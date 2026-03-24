import os
import re
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def remove_pertinent_results(text: str) -> str:
    text = re.sub(
        r"pertinent results:.*?(brief hospital course:)",
        r"\1",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )
    return text

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # remove de-identification placeholders
    text = text.replace("___", " ")

    # remove line breaks/tabs
    text = text.replace("\n", " ").replace("\t", " ")

    # remove explicit clock times like 10:25pm
    text = re.sub(r"\b\d{1,2}:\d{2}(am|pm)\b", " ", text)

    # remove standalone dates/times fragments if desired
    text = re.sub(r"\b\d{1,2}(am|pm)\b", " ", text)

    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    text = remove_pertinent_results(text)

    return text

def normalize_label(x):
    x = str(x).strip().lower()
    if x == "yes":
        return 1
    elif x == "no":
        return 0
    else:
        return None


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    notes_df = pd.read_csv(args.notes_csv)
    readm_df = pd.read_csv(args.readmission_csv)

    # keep only discharge summaries if needed
    if "note_type" in notes_df.columns:
        notes_df = notes_df[notes_df["note_type"] == "DS"].copy()

    # keep needed note columns
    needed_note_cols = [c for c in [
        "note_id", "subject_id", "hadm_id", "note_type",
        "note_seq", "charttime", "storetime", "text", "token_count"
    ] if c in notes_df.columns]
    notes_df = notes_df[needed_note_cols].copy()

    # clean text
    notes_df["clean_text"] = notes_df["text"].apply(clean_text)

    # normalize label
    label_col = args.label_col
    readm_df[label_col] = readm_df[label_col].apply(normalize_label)

    # keep only rows with valid labels
    readm_df = readm_df.dropna(subset=[label_col]).copy()
    readm_df[label_col] = readm_df[label_col].astype(int)

    # merge on subject_id
    merged_df = notes_df.merge(readm_df, on="subject_id", how="inner")

    # drop empty text
    merged_df = merged_df[merged_df["clean_text"].str.len() > 0].reset_index(drop=True)

    # keep final columns
    final_cols = [c for c in [
        "note_id", "subject_id", "hadm_id", "note_type",
        "note_seq", "charttime", "storetime", "token_count",
        "clean_text", label_col,
        "first_visit", "first_visit_length_days",
        "second_visit", "second_visit_length_days", "interval_days"
    ] if c in merged_df.columns]

    merged_df = merged_df[final_cols].copy()

    if len(merged_df) < 10:
        raise ValueError("Dataset is too small after merging to safely split.")

    # train 80%, temp 20%
    train_df, temp_df = train_test_split(
        merged_df,
        test_size=0.2,
        random_state=args.seed,
        stratify=merged_df[label_col]
    )

    # split temp into val 10%, test 10%
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=args.seed,
        stratify=temp_df[label_col]
    )

    train_path = os.path.join(args.output_dir, "train.csv")
    val_path = os.path.join(args.output_dir, "val.csv")
    test_path = os.path.join(args.output_dir, "test.csv")
    merged_path = os.path.join(args.output_dir, "all_merged.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    merged_df.to_csv(merged_path, index=False)

    print(f"Saved merged data: {merged_path}")
    print(f"Saved train: {train_path} ({len(train_df)} rows)")
    print(f"Saved val:   {val_path} ({len(val_df)} rows)")
    print(f"Saved test:  {test_path} ({len(test_df)} rows)")
    print("\nLabel distribution:")
    print("Train:")
    print(train_df[label_col].value_counts(normalize=True))
    print("Val:")
    print(val_df[label_col].value_counts(normalize=True))
    print("Test:")
    print(test_df[label_col].value_counts(normalize=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--notes_csv", type=str, required=True,
                        help="Path to discharge notes CSV")
    parser.add_argument("--readmission_csv", type=str, required=True,
                        help="Path to readmission labels CSV")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--label_col", type=str, default="30_day_readmission")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)