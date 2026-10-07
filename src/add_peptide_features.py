"""Add peptide sequence features to the TCR-peptide dataset."""

from pathlib import Path

import pandas as pd

from src.sequence_features import build_peptide_features


BASE = Path(__file__).resolve().parent.parent
SOURCE_CSV = BASE / "processed" / "data_tcren_score.csv"
FINAL_CSV = BASE / "processed" / "data_tcren_features.csv"


# Load scored TCR-peptide pairs
df = pd.read_csv(SOURCE_CSV)

required_cols = ["pdb_id", "peptide", "label", "tcren_score"]
missing = [col for col in required_cols if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df["pdb_id"] = df["pdb_id"].astype(str).str.strip().str.lower()
df["peptide"] = df["peptide"].astype(str).str.strip().str.upper()


# Generate peptide features using the same function used at inference time
feature_df = pd.DataFrame(
    df["peptide"].apply(build_peptide_features).tolist()
)

df_out = pd.concat(
    [df.reset_index(drop=True), feature_df],
    axis=1,
)


# Save processed dataset
df_out.to_csv(FINAL_CSV, index=False)

print(f"Saved to: {FINAL_CSV.resolve()}")
