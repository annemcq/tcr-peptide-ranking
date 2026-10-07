"""Add CDR3 sequence features to the TCR-peptide dataset."""

from pathlib import Path

import pandas as pd

from src.sequence_features import build_tcr_features


BASE = Path(__file__).resolve().parent.parent
DATA_CSV = BASE / "processed" / "data_tcren_features.csv"
SUMMARY_CSV = BASE / "data" / "summary_PDB_structures.csv"
OUT_CSV = BASE / "processed" / "data_tcren_features_with_tcr.csv"


# Load peptide-feature dataset and TCR sequence information
df = pd.read_csv(DATA_CSV)
summary = pd.read_csv(SUMMARY_CSV)

df["pdb_id"] = df["pdb_id"].astype(str).str.strip().str.lower()
summary["pdb.id"] = summary["pdb.id"].astype(str).str.strip().str.lower()

for col in ["cdr3a", "cdr3b"]:
    summary[col] = summary[col].astype(str).str.strip().str.upper()


# Keep one CDR3 alpha/beta pair per TCR
summary_small = (
    summary[["pdb.id", "cdr3a", "cdr3b"]]
    .drop_duplicates("pdb.id")
    .copy()
)


# Generate TCR features using the same function used at inference time
tcr_features = summary_small.apply(
    lambda row: build_tcr_features(
        row["cdr3a"],
        row["cdr3b"],
    ),
    axis=1,
)

feature_df = pd.DataFrame(tcr_features.tolist(), index=summary_small.index)

summary_small = pd.concat(
    [summary_small, feature_df],
    axis=1,
)

summary_small = summary_small.rename(
    columns={"pdb.id": "pdb_id"}
)


# Merge TCR features with the peptide-feature dataset
df_out = df.merge(
    summary_small,
    on="pdb_id",
    how="left",
)

missing_rows = (
    df_out["cdr3a"].isna().sum()
    + df_out["cdr3b"].isna().sum()
)

print("Rows with missing CDR3 info:", missing_rows)


# Save final feature dataset
df_out.to_csv(OUT_CSV, index=False)

print("Input shape:", df.shape)
print("Output shape:", df_out.shape)
print("Saved to:", OUT_CSV)
