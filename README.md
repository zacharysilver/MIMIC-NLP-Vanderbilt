# How to run

# Step 1: Preprocess and Merge
python preprocess_data.py --notes_csv data/discharge_notes.csv --readmission_csv data/readmission.csv --output_dir data

# Step 2: Run LSA Analysis
python main.py --train_csv data/train.csv --val_csv data/val.csv --test_csv data/test.csv
# Step 3: Run Transformer-based analysis
python frozenBert.py
# Required Packages
conda create -n HNLP_project_env python=3.10 -y && conda activate HNLP_project_env && pip install -r requirements.txt
