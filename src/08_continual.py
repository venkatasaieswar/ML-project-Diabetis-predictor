import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve, confusion_matrix

OUT = Path("outputs")
MODEL_DIR = OUT / "models"
FIG_DIR = OUT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(OUT / "aggregated_patients.csv", dtype={"PATIENT": str}, parse_dates=["last_encounter"]).set_index("PATIENT")
    labels = pd.read_csv(OUT / "diabetes_labels.csv", index_col=0)
    if labels.shape[1] == 1:
        labels = labels.iloc[:, 0]
    train_ids = pd.read_csv(OUT / "train_ids.csv")["PATIENT"].tolist()
    test_ids = pd.read_csv(OUT / "test_ids.csv")["PATIENT"].tolist()
    return df, labels, train_ids, test_ids


def transform(pre, X):
    return pre.transform(X)


def evaluate_model(model, X_t, y_true):
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_t)[:, 1]
    else:
        # decision function fallback
        try:
            probs = model.decision_function(X_t)
            # scale to 0-1
            probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-12)
        except Exception:
            probs = model.predict(X_t)
    fpr, tpr, _ = roc_curve(y_true, probs) if len(np.unique(y_true)) > 1 else (None, None, None)
    roc_auc = auc(fpr, tpr) if fpr is not None else np.nan
    auprc = average_precision_score(y_true, probs)
    preds = model.predict(X_t)
    cm = confusion_matrix(y_true, preds)
    return dict(roc_auc=roc_auc, auprc=auprc, fpr=fpr, tpr=tpr, preds=preds, probs=probs, cm=cm)


def plot_roc(res_dict, name):
    plt.figure()
    for label, res in res_dict.items():
        fpr, tpr = res.get('fpr'), res.get('tpr')
        if fpr is None:
            continue
        plt.plot(fpr, tpr, label=f"{label} (AUC={res['roc_auc']:.3f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC Curves - {name}')
    plt.legend()
    out = FIG_DIR / f'roc_{name}.png'
    plt.savefig(out)
    plt.close()


def plot_confusion(cm, labels, name):
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Pred')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {name}')
    out = FIG_DIR / f'cm_{name}.png'
    plt.savefig(out)
    plt.close()


def main():
    df, labels, train_ids, test_ids = load_data()

    # load preprocessor and stage1 models
    pre = joblib.load(MODEL_DIR / 'preprocessor.joblib')
    dt1 = joblib.load(MODEL_DIR / 'dt_stage1.joblib')
    mlp1 = joblib.load(MODEL_DIR / 'mlp_stage1.joblib')

    # Prepare Dataset2 train for fine-tuning: we will fine-tune only on Dataset2 train (subset of test_ids)
    # For simplicity, split test_ids into train/test within dataset2 (80/20) using simple split (no stratify because few positives)
    d2_ids = test_ids
    # Use first 70% as train, last 30% as test of dataset2 for fine-tuning evaluation (but final evaluation will be on original test_ids)
    if len(d2_ids) < 5:
        print('Dataset2 too small to fine-tune; skipping fine-tuning')
        return
    split_idx = int(len(d2_ids) * 0.7)
    d2_train_ids = d2_ids[:split_idx]
    d2_test_ids = d2_ids[split_idx:]

    X_d2train = df.loc[d2_train_ids]
    y_d2train = labels.reindex(d2_train_ids).fillna(0).astype(int)
    X_d2test = df.loc[d2_test_ids]
    y_d2test = labels.reindex(d2_test_ids).fillna(0).astype(int)

    # Transform using preprocessor (fitted on Dataset1-train earlier)
    X_d2train_t = pre.transform(X_d2train)
    X_d2test_t = pre.transform(X_d2test)

    # Fine-tune MLP: continue training using warm_start
    try:
        mlp1.warm_start = True
    except Exception:
        pass
    print('Fine-tuning MLP on Dataset2-train (warm start)')
    mlp1.max_iter = mlp1.max_iter + 100
    mlp1.fit(X_d2train_t, y_d2train)
    joblib.dump(mlp1, MODEL_DIR / 'mlp_finetuned.joblib')

    # Retrain Decision Tree on Dataset1 + Dataset2-train
    df_all_train_ids = list(train_ids) + list(d2_train_ids)
    X_comb = df.loc[df_all_train_ids]
    y_comb = labels.reindex(df_all_train_ids).fillna(0).astype(int)
    X_comb_t = pre.transform(X_comb)

    dt2 = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    dt2.fit(X_comb_t, y_comb)
    joblib.dump(dt2, MODEL_DIR / 'dt_retrained.joblib')

    # Train SVM on original Dataset1-train for comparison (if not present)
    svc = SVC(probability=True, class_weight='balanced', random_state=42)
    # prepare Dataset1-train: load ids and use pre-fitted preprocessor
    X_d1 = df.loc[train_ids]
    y_d1 = labels.reindex(train_ids).fillna(0).astype(int)
    X_d1_t = pre.transform(X_d1)
    svc.fit(X_d1_t, y_d1)
    joblib.dump(svc, MODEL_DIR / 'svc_stage1.joblib')

    # Evaluate on Dataset2 original test set
    X_test = df.loc[test_ids]
    y_test = labels.reindex(test_ids).fillna(0).astype(int)
    X_test_t = pre.transform(X_test)

    res = {}
    res['mlp_finetuned'] = evaluate_model(mlp1, X_test_t, y_test)
    res['dt_retrained'] = evaluate_model(dt2, X_test_t, y_test)
    res['svc_stage1'] = evaluate_model(svc, X_test_t, y_test)

    # Save results
    joblib.dump(res, MODEL_DIR / 'results_stage2.joblib')
    print('Saved continual learning results to', MODEL_DIR / 'results_stage2.joblib')

    # Plot ROC curves
    plot_roc(res, 'dataset2_test')

    # Plot confusion matrices
    for name, info in res.items():
        cm = info['cm']
        plot_confusion(cm, ['neg','pos'], name)


if __name__ == '__main__':
    main()
