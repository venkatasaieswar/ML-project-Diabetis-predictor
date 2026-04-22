
#  Clinical Prediction Dashboard 

This repository contains the code, dashboard, and pipeline scripts for the Diabetes prediction assignment under temporal shift.

Important: The raw CSV dataset is large and should NOT be committed. Keep the CSV files locally in a folder named `csv/` (same filenames as the original dataset). The pipeline scripts read from `csv/` and write results into `outputs/`.

Minimum requirements

- Python 3.10+ (3.12 recommended)
- 2 GB free disk space for intermediate files

Recommended dependencies (install in a virtual environment):

1. Create and activate a virtual environment:

   python3 -m venv .venv
   source .venv/bin/activate

2. Install required Python packages:

   pip install pandas numpy scikit-learn imbalanced-learn streamlit rapidfuzz joblib altair matplotlib seaborn python-dateutil

   (Alternatively, if you maintain a requirements.txt for your environment, use `pip install -r requirements.txt`.)

Pipeline: how to run (single, explicit flow)

1. Place the dataset CSV files in the repository root under the `csv/` directory. Expected files include (but are not limited to):
   - patients.csv, encounters.csv, observations.csv, medications.csv, conditions.csv

2. Run the data aggregation step (creates patient-level features):

   python3 src/04_aggregate.py

3. Create train/test splits and compute labels:

   python3 src/05_split_and_label.py

4. Train baseline models and save artifacts:

   python3 src/07_train.py

5. Run continual learning / stage-2 training (fine-tune / retrain):

   python3 src/08_continual.py

6. Compute explainability artifacts and EDA plots used by the dashboard:

   python3 src/09_explain_eda.py

7. (Optional) Dump joblib summaries for inspection:

   python3 src/10_dump_joblib.py

Running the dashboard

After completing the pipeline, start the Streamlit dashboard which reads precomputed artifacts from `outputs/`:

   streamlit run TeamXX_Assignment2_dashboard.py

Alternative: using uv

If you prefer the lightweight `uv` wrapper, here are equivalent commands using `uv run`.

1. Install uv (optional):

   pip install uv

2. Run the pipeline with uv (example sequence):

   uv run python3 src/04_aggregate.py
   uv run python3 src/05_split_and_label.py
   uv run python3 src/07_train.py
   uv run python3 src/08_continual.py
   uv run python3 src/09_explain_eda.py
   uv run python3 src/10_dump_joblib.py

3. Start the dashboard with uv:

   uv run streamlit run TeamXX_Assignment2_dashboard.py

What to commit to Git

- Do commit: source code in `src/`, the dashboard script, README, and lightweight outputs (models and small figures) if desired.
- Do NOT commit: the raw CSV dataset in `csv/` (1.5 GB in this project). `.gitignore` is configured to ignore `csv/`.

Repository layout

- csv/            # raw CSVs (local only; not committed)
- outputs/        # artifacts written by the pipeline (models, figures, summaries)
- src/            # data preparation and training scripts
- TeamXX_Assignment2_dashboard.py  # Streamlit dashboard (single-file)
- README.md
- pyproject.toml

Troubleshooting

- If a script fails due to parsing (large CSVs), rerun the script with increased memory or ensure you have the right CSV files in `csv/`.
- If Streamlit raises widget key errors, make sure you are using the updated dashboard file in this repo.

Contact and authors

Team members: Member 1, Member 2, Member 3, Member 4

If you want, I can also prepare a small ZIP with selected outputs and the dashboard file for submission.
