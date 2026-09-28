# FEATURE IMPORTANCE PLOT (RANDOM FOREST)

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

# Required settings
BASE = Path(__file__).resolve().parent.parent
source_csv = BASE / "processed" / "data_tcren_features_with_tcr.csv"
out_png = BASE / "plots" / "feature_importance_rf_full.png"
out_csv = BASE / "results" / "feature_importance_rf_full.csv"

###############################################################
# Load data

df = pd.read_csv(source_csv)
df["pdb_id"] = df["pdb_id"].astype(str).str.strip().str.lower()
df["peptide"] = df["peptide"].astype(str).str.strip().str.upper()

# Select features

aa_cols = [c for c in df.columns if c.startswith("frac_")]

pep_cols = [
    "length",
    "hydrophobic_frac",
    "polar_frac",
    "positive_frac",
    "negative_frac",
    "aromatic_frac",
    "small_frac",
    "tiny_frac",
    "proline_frac",
    "glycine_frac",
    "mw_est",
] + aa_cols

tcr_cols = [
    "cdr3a_len",
    "cdr3b_len",
    "cdr3a_hydrophobic_frac",
    "cdr3a_positive_frac",
    "cdr3a_negative_frac",
    "cdr3a_aromatic_frac",
    "cdr3a_glycine_frac",
    "cdr3a_proline_frac",
    "cdr3b_hydrophobic_frac",
    "cdr3b_positive_frac",
    "cdr3b_negative_frac",
    "cdr3b_aromatic_frac",
    "cdr3b_glycine_frac",
    "cdr3b_proline_frac",
]

feature_cols = pep_cols + tcr_cols + ["tcren_score"]

missing = [c for c in feature_cols + ["label"] if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")


X = df[feature_cols]
y = df["label"]

###############################################################
# Random Forest

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X, y)

importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

importance_df.to_csv(out_csv, index=False)

top_n = 15
top_df = importance_df.head(top_n).iloc[::-1]

###############################################################

# Plot
plt.figure(figsize=(8, 6))
plt.barh(top_df["feature"], top_df["importance"])
plt.xlabel("Feature importance")
plt.title("Top 15 feature importances (RF full)")
plt.tight_layout()
plt.savefig(out_png, dpi=300)
plt.close()

###############################################################

# Save
print("Saved:")
print(out_csv)
print(out_png)