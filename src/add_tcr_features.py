# TCR FEATURE AUGMENTATION

# Import necessary libraries
import pandas as pd
from pathlib import Path

# Required settings
BASE = Path(__file__).resolve().parent.parent
data_csv = BASE / "processed" / "data_tcren_features.csv"
summary_csv = BASE / "data" / "summary_PDB_structures.csv"
out_csv = BASE / "processed" / "data_tcren_features_with_tcr.csv"

df = pd.read_csv(data_csv)
summary = pd.read_csv(summary_csv)

df["pdb_id"] = df["pdb_id"].astype(str).str.strip().str.lower()
summary["pdb.id"] = summary["pdb.id"].astype(str).str.strip().str.lower()

###############################################################
# TCR features

for col in ["cdr3a", "cdr3b"]:
    summary[col] = summary[col].astype(str).str.strip().str.upper()

hydrophobic = set("AVILMFWY")
positive = set("KRH")
negative = set("DE")
aromatic = set("FWYH")

###############################################################
# Define helper functions
def frac_in_set(seq, aa_set):
    if not seq:
        return 0.0
    return sum(aa in aa_set for aa in seq) / len(seq)

def aa_frac(seq, aa):
    if not seq:
        return 0.0
    return seq.count(aa) / len(seq)

###############################################################
# Additional TCR features in data 

summary_small = summary[["pdb.id", "cdr3a", "cdr3b"]].drop_duplicates("pdb.id").copy()

# Compute length features
summary_small["cdr3a_len"] = summary_small["cdr3a"].str.len()
summary_small["cdr3b_len"] = summary_small["cdr3b"].str.len()
# Composition features
summary_small["cdr3a_hydrophobic_frac"] = summary_small["cdr3a"].apply(lambda x: frac_in_set(x, hydrophobic))
summary_small["cdr3a_positive_frac"] = summary_small["cdr3a"].apply(lambda x: frac_in_set(x, positive))
summary_small["cdr3a_negative_frac"] = summary_small["cdr3a"].apply(lambda x: frac_in_set(x, negative))
summary_small["cdr3a_aromatic_frac"] = summary_small["cdr3a"].apply(lambda x: frac_in_set(x, aromatic))
summary_small["cdr3a_glycine_frac"] = summary_small["cdr3a"].apply(lambda x: aa_frac(x, "G"))
summary_small["cdr3a_proline_frac"] = summary_small["cdr3a"].apply(lambda x: aa_frac(x, "P"))
# Repeat for beta chain
summary_small["cdr3b_hydrophobic_frac"] = summary_small["cdr3b"].apply(lambda x: frac_in_set(x, hydrophobic))
summary_small["cdr3b_positive_frac"] = summary_small["cdr3b"].apply(lambda x: frac_in_set(x, positive))
summary_small["cdr3b_negative_frac"] = summary_small["cdr3b"].apply(lambda x: frac_in_set(x, negative))
summary_small["cdr3b_aromatic_frac"] = summary_small["cdr3b"].apply(lambda x: frac_in_set(x, aromatic))
summary_small["cdr3b_glycine_frac"] = summary_small["cdr3b"].apply(lambda x: aa_frac(x, "G"))
summary_small["cdr3b_proline_frac"] = summary_small["cdr3b"].apply(lambda x: aa_frac(x, "P"))

summary_small = summary_small.rename(columns={"pdb.id": "pdb_id"})

# Merge with original dataset
df_out = df.merge(summary_small, on="pdb_id", how="left")

missing_rows = df_out["cdr3a"].isna().sum() + df_out["cdr3b"].isna().sum()
print("Rows with missing CDR3 info:", missing_rows)

###############################################################
# Save
df_out.to_csv(out_csv, index=False)

print("Input shape:", df.shape)
print("Output shape:", df_out.shape)
print("Saved to:", out_csv)