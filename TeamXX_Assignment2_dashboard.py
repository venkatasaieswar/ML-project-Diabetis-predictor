import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import altair as alt
from datetime import datetime
from typing import List

# fuzzy match
from rapidfuzz import process

# Paths
OUT = Path('outputs')
MODELS = OUT / 'models'
FIGS = OUT / 'figures'
REPORTS = OUT / 'reports'

st.set_page_config(layout='wide', page_title='TeamXX Assignment2 Dashboard')

# Team placeholders
TEAM_NAME = 'TeamXX'
TEAM_MEMBERS = ['Member 1', 'Member 2', 'Member 3', 'Member 4']

# color palette (peaceful)
PALETTE = {
    'dataset1': '#2a9d8f',
    'dataset2': '#f4a261',
    'accent': '#264653'
}


@st.cache_data
def load_aggregated():
    p = OUT / 'aggregated_patients.csv'
    if not p.exists():
        return None
    df = pd.read_csv(p, dtype={'PATIENT':str}, parse_dates=['last_encounter'])
    if 'PATIENT' in df.columns:
        df = df.set_index('PATIENT')
    return df


@st.cache_data
def load_labels():
    p = OUT / 'diabetes_labels.csv'
    if not p.exists():
        return None
    lab = pd.read_csv(p, index_col=0)
    if lab.shape[1] == 1:
        lab = lab.iloc[:,0]
    return lab


@st.cache_resource
def load_models_and_artifacts():
    artifacts = {}
    # preprocessor
    pre = None
    try:
        pre = joblib.load(MODELS / 'preprocessor.joblib')
    except Exception:
        pre = None
    artifacts['preprocessor'] = pre
    # models
    for name in ['dt_retrained.joblib','mlp_finetuned.joblib','svc_stage1.joblib']:
        p = MODELS / name
        if p.exists():
            try:
                artifacts[name] = joblib.load(p)
            except Exception:
                artifacts[name] = None
    # metrics and summaries
    metrics = None
    ms = None
    if (REPORTS/'models_metrics.csv').exists():
        metrics = pd.read_csv(REPORTS/'models_metrics.csv')
    if (REPORTS/'models_summary.json').exists():
        import json
        ms = json.load(open(REPORTS/'models_summary.json'))
    artifacts['metrics'] = metrics
    artifacts['models_summary'] = ms
    return artifacts


def predict_for_patient(patient_id, df, pre, models):
    if patient_id not in df.index:
        return {}
    x = df.loc[[patient_id]]
    # drop label if present
    if 'label' in x.columns:
        x = x.drop(columns=['label'])
    # transform
    try:
        X_t = pre.transform(x)
    except Exception:
        X_t = None
    out = {}
    for key, m in models.items():
        if key not in ['preprocessor','metrics','models_summary'] and m is not None:
            try:
                if X_t is not None and hasattr(m, 'predict_proba'):
                    prob = float(m.predict_proba(X_t)[:,1][0])
                elif X_t is not None and hasattr(m, 'decision_function'):
                    val = m.decision_function(X_t)[0]
                    prob = 1/(1+np.exp(-val))
                else:
                    prob = None
            except Exception:
                prob = None
            out[key] = prob
    return out


def fuzzy_patient_search(query, choices, limit=50):
    if not query:
        return list(choices)[:limit]
    results = process.extract(query, choices, limit=limit)
    # results: list of (choice, score, idx)
    return [r[0] for r in results]


def main():
    # Header
    st.markdown(f"# Clinical Prediction Dashboard — {TEAM_NAME}")
    st.markdown(f"**Team members:** {', '.join(TEAM_MEMBERS)}")
    st.markdown('---')

    # Load data
    df = load_aggregated()
    labels = load_labels()
    artifacts = load_models_and_artifacts()
    pre = artifacts.get('preprocessor')

    if df is None or labels is None:
        st.error('Required outputs not found in outputs/. Please run the preprocessing pipeline first.')
        return

    # Basic numbers
    total = len(df)
    total_pos = int(labels.sum())
    train_ids = pd.read_csv(OUT/'train_ids.csv')['PATIENT'].tolist() if (OUT/'train_ids.csv').exists() else []
    test_ids = pd.read_csv(OUT/'test_ids.csv')['PATIENT'].tolist() if (OUT/'test_ids.csv').exists() else []

    # KPI cards
    k1,k2,k3,k4 = st.columns([1,1,1,1])
    k1.metric('Total patients', f'{total}')
    k2.metric('Diabetes positives', f'{total_pos}', f"{total_pos/total:.1%}")
    k3.metric('Historical (Dataset1)', f'{len(train_ids)}')
    k4.metric('Current (Dataset2)', f'{len(test_ids)}')

    st.markdown('---')

    tabs = st.tabs(['EDA','Data Analysis','Models','Patient Inspector','About'])

    # EDA tab (first)
    with tabs[0]:
        st.header('Exploratory Data Analysis (EDA)')
        st.write('Sequential comparison of numeric features across Dataset1 (historical) and Dataset2 (current).')
        numeric_candidates = ['Body Weight_last','BMI_last','Glucose_last','med_count','cond_count']
        numeric_present = [c for c in numeric_candidates if c in df.columns]
        if not numeric_present:
            st.info('No numeric features found in the aggregated patient table to display.')
        else:
            sdf = df.copy()
            sdf['dataset'] = np.where(sdf.index.isin(train_ids), 'Dataset1', 'Dataset2')
            for feat in numeric_present:
                st.subheader(feat)
                # density chart across datasets
                try:
                    chart = alt.Chart(sdf.reset_index()).transform_density(feat, as_=[feat, 'density'], groupby=['dataset']).mark_area(opacity=0.45).encode(x=feat+':Q', y='density:Q', color='dataset:N').properties(height=220)
                    st.altair_chart(chart, use_container_width=True)
                except Exception:
                    st.write('Could not render density for this feature.')
                # stats table for the feature
                d1 = sdf[sdf['dataset']=='Dataset1'][feat].dropna()
                d2 = sdf[sdf['dataset']=='Dataset2'][feat].dropna()
                stats_rows = []
                stats_rows.append({'dataset':'Dataset1','count':len(d1),'mean':float(d1.mean()) if len(d1)>0 else np.nan,'median':float(d1.median()) if len(d1)>0 else np.nan,'std':float(d1.std()) if len(d1)>0 else np.nan})
                stats_rows.append({'dataset':'Dataset2','count':len(d2),'mean':float(d2.mean()) if len(d2)>0 else np.nan,'median':float(d2.median()) if len(d2)>0 else np.nan,'std':float(d2.std()) if len(d2)>0 else np.nan})
                st.table(pd.DataFrame(stats_rows).set_index('dataset').style.format({'mean':'{:.3f}','median':'{:.3f}','std':'{:.3f}'}))
                # percent change interpretation
                mean1 = stats_rows[0]['mean']
                mean2 = stats_rows[1]['mean']
                if not np.isnan(mean1) and not np.isnan(mean2) and mean1!=0:
                    pct = (mean2-mean1)/mean1
                    arrow = 'increased' if pct>0 else 'decreased' if pct<0 else 'no change'
                    st.write(f'Short interpretation: mean {feat} has {arrow} by {pct:.1%} from Dataset1 to Dataset2.')
                else:
                    st.write('Short interpretation: insufficient data to compute percent change for this feature.')

    # Data Analysis tab (second) - moved prevalence and summary here
    with tabs[1]:
        st.header('Data Analysis')
        st.write('High-level dataset summaries and prevalence over time. This tab contains the prevalence chart and top-level interpretation.')
        # prevalence over time (interactive)
        df2 = df.copy()
        df2['label'] = labels.reindex(df2.index).fillna(0).astype(int)
        df2['ym'] = df2['last_encounter'].dt.to_period('M').dt.to_timestamp()
        prevalence = df2.groupby('ym')['label'].mean().reset_index()
        prevalence.columns = ['month','prevalence']
        chart = alt.Chart(prevalence).mark_line(point=True).encode(x='month:T', y=alt.Y('prevalence:Q', axis=alt.Axis(format='%'))).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
        # Short interpretation summary (concise)
        st.markdown('**Quick summary**')
        col_a, col_b = st.columns([2,3])
        with col_a:
            st.metric('Total patients', f'{total}')
            st.metric('Diabetes positives', f'{total_pos}', f"{total_pos/total:.1%}")
        with col_b:
            st.write('- Historical (Dataset1): ' + (f"{len(train_ids)} patients" if train_ids else 'N/A'))
            st.write('- Current (Dataset2): ' + (f"{len(test_ids)} patients" if test_ids else 'N/A'))
            st.write('\n')
            st.write('Interpretation: Prevalence is low (~{:.1%}). Models were developed on Dataset1 and tested on Dataset2 to assess temporal generalization; consult the Models tab for performance summaries and EDA for per-feature comparisons.'.format(total_pos/total if total>0 else 0))

    # EDA & Drift
    with tabs[1]:
        st.header('Exploratory Data Analysis & Drift')
        st.write('Compare distributions between historical and current datasets.')
        numeric_candidates = ['Body Weight_last','BMI_last','Glucose_last','med_count','cond_count']
        numeric_present = [c for c in numeric_candidates if c in df.columns]
        col1, col2 = st.columns(2)
        with col1:
            feat = st.selectbox('Select feature', numeric_present)
            # build density chart
            sdf = df.copy()
            sdf['dataset'] = np.where(sdf.index.isin(train_ids), 'Dataset1', 'Dataset2')
            chart = alt.Chart(sdf.reset_index()).transform_density(feat, as_=[feat, 'density'], groupby=['dataset']).mark_area(opacity=0.45).encode(x=feat+':Q', y='density:Q', color='dataset:N').properties(height=300)
            st.altair_chart(chart, use_container_width=True)
        with col2:
            st.write('Summary statistics')
            stats = df[numeric_present].describe().T[['mean','50%','std']]
            st.dataframe(stats)
        # Cross-dataset comparison table (compact)
        if numeric_present:
            st.markdown('**Cross-dataset numeric comparison**')
            sdf = df.copy()
            sdf['dataset'] = np.where(sdf.index.isin(train_ids), 'Dataset1', 'Dataset2')
            rows = []
            for f in numeric_present:
                d1 = sdf[sdf['dataset']=='Dataset1'][f].dropna()
                d2 = sdf[sdf['dataset']=='Dataset2'][f].dropna()
                mean1 = float(d1.mean()) if len(d1)>0 else float('nan')
                mean2 = float(d2.mean()) if len(d2)>0 else float('nan')
                med1 = float(d1.median()) if len(d1)>0 else float('nan')
                med2 = float(d2.median()) if len(d2)>0 else float('nan')
                std1 = float(d1.std()) if len(d1)>0 else float('nan')
                std2 = float(d2.std()) if len(d2)>0 else float('nan')
                pct_change = (mean2-mean1)/mean1 if mean1 and not np.isnan(mean1) else float('nan')
                rows.append({'feature':f,'mean_dataset1':mean1,'mean_dataset2':mean2,'median_dataset1':med1,'median_dataset2':med2,'std_dataset1':std1,'std_dataset2':std2,'pct_change_mean':pct_change})
            comp = pd.DataFrame(rows).set_index('feature')
            st.dataframe(comp.style.format({c: '{:.3f}' for c in comp.columns if 'pct_change' not in c}).format({'pct_change_mean':'{:.1%}'}))
            # interpretation for selected feature
            if feat in comp.index:
                r = comp.loc[feat]
                ch = r['pct_change_mean']
                if np.isnan(ch):
                    st.write('No sufficient data to compare this feature across datasets.')
                else:
                    arrow = 'increased' if ch>0 else 'decreased' if ch<0 else 'no change'
                    st.write(f'Short interpretation: {feat} has {arrow} by {ch:.1%} from Dataset1 to Dataset2 (mean). Consider how this shift may impact model generalization.)')

    # Models tab
    with tabs[2]:
        st.header('Models & Performance')
        metrics = artifacts.get('metrics')
        if metrics is not None:
            st.subheader('Metrics table')
            st.dataframe(metrics.style.highlight_max(subset=['auprc']))
            st.download_button('Download metrics CSV', data=metrics.to_csv(index=False).encode(), file_name='models_metrics.csv', key='metrics_table_download')
            # KPI highlights
            try:
                best = metrics.loc[metrics['auprc'].idxmax()]
                k1,k2,k3 = st.columns([1,1,2])
                k1.metric('Best AUPRC', f"{best['auprc']:.3f}")
                k2.metric('Best model', f"{best['model']}")
                k3.write(f"Stage: {best.get('stage','N/A')} — short note: this is the highest AUPRC observed. Check confusion matrices and ROC for class tradeoffs.")
            except Exception:
                pass
            # bar chart of AUPRC
            chart = alt.Chart(metrics).mark_bar().encode(x='model:N', y='auprc:Q', color=alt.Color('stage:N', scale=alt.Scale(range=[PALETTE['dataset1'], PALETTE['dataset2']])))
            st.altair_chart(chart, use_container_width=True)
        # show ROC image if present
        if (FIGS/'roc_dataset2_test.png').exists():
            st.image(str(FIGS/'roc_dataset2_test.png'), caption='ROC Curves on Dataset2')
        st.write('Confusion matrices:')
        cm_files = sorted([p for p in FIGS.glob('cm_*.png')])
        for p in cm_files:
            st.image(str(p))
        # Feature importance images (if available)
        fi_dt = FIGS/'fi_dt_retrained.png'
        fi_mlp = FIGS/'fi_mlp_permutation.png'
        if fi_dt.exists() or fi_mlp.exists():
            st.subheader('Feature importance')
            cols = st.columns(2)
            if fi_dt.exists():
                cols[0].image(str(fi_dt), caption='Decision Tree feature importance')
            if fi_mlp.exists():
                cols[1].image(str(fi_mlp), caption='MLP permutation importance')
            st.write('Short interpretation: Feature importance plots show which variables the models rely on most; compare DT importance (direct) with MLP permutation (sensitivity-based). Large differences suggest model class-dependence on features.')

    # Patient inspector
    with tabs[3]:
        st.header('Patient Inspector')
        st.write('Search for a patient by ID (fuzzy search supported).')
        all_ids = list(df.index)
        query = st.text_input('Patient ID or partial')
        matches = fuzzy_patient_search(query, all_ids, limit=200)
        pid = st.selectbox('Select patient', options=matches)
        if pid:
            row = df.loc[pid]
            st.subheader('Key patient info')
            info = {k: row[k] if k in row.index else None for k in ['age_at_last','GENDER','RACE','ZIP','last_encounter']}
            st.json(info)
            # key features
            feats = ['Body Weight_last','BMI_last','Glucose_last','med_count','cond_count']
            present_feats = {f: row[f] for f in feats if f in row.index}
            st.metric('Risk estimate (models)', '')
            preds = predict_for_patient(pid, df, pre, artifacts)
            cols = st.columns(len(preds) if preds else 1)
            for i,(k,v) in enumerate(preds.items()):
                pct = f"{v*100:.1f}%" if v is not None else 'N/A'
                cols[i].metric(k.replace('.joblib',''), pct)
            st.write('Patient features:')
            st.table(pd.DataFrame.from_dict(present_feats, orient='index', columns=['value']))

    # About tab
    with tabs[4]:
        st.header('About & Notes')
        st.write('This dashboard is a read-only viewer of precomputed models and analyses. For advanced users, expand the developer panel below.')
        st.markdown('**Interpretation & Conclusions**')
        st.write(f'- Total patients: {total}')
        st.write(f'- Diabetes positives: {total_pos} ({total_pos/total:.1%} prevalence)' if total>0 else f'- Diabetes positives: {total_pos}')
        st.write(f'- Historical (Dataset1): {len(train_ids)} patients')
        st.write(f'- Current (Dataset2): {len(test_ids)} patients')
        if metrics is not None:
            try:
                best = metrics.loc[metrics['auprc'].idxmax()]
                st.write(f"- Best performing model: {best['model']} (AUPRC={best['auprc']:.3f}, stage={best.get('stage','N/A')})")
                st.write('- General guidance: If performance drops from Dataset1->Dataset2, consider retraining or continual learning (we provide fine-tuned/retrained artifacts).')
            except Exception:
                pass
        with st.expander('Developer / Advanced'):
            st.write('Artifacts location: outputs/ folder. Models and metrics are available for download.')
            if (REPORTS/'models_summary.json').exists():
                with open(REPORTS/'models_summary.json') as fh:
                    st.code('\n'.join(list(fh)[:50]))
            if (REPORTS/'models_metrics.csv').exists():
                st.download_button('Download metrics CSV', data=open(REPORTS/'models_metrics.csv','rb').read(), file_name='models_metrics.csv', key='metrics_dev_download')

    # Footer
    st.markdown('---')
    st.caption(f'Last updated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")} | Team: {TEAM_NAME}')


if __name__ == '__main__':
    main()
