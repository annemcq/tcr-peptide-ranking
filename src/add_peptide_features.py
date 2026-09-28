# PEPTIDE FEATURE GENERATION

import pandas as pd
from pathlib import Path

# Baseline settings 

BASE = Path(__file__).resolve().parent.parent
Source_CSV = BASE / "processed" / "data_tcren_score.csv"
Final_CSV = BASE / "processed" / "data_tcren_features.csv"

AA = list("ACDEFGHIKLMNPQRSTVWY")
# Molecular weight of aminoacids
AA_mol_weight = {
    "A": 89.09, "C": 121.16, "D": 133.10, "E": 147.13, "F": 165.19,
    "G": 75.07, "H": 155.16, "I": 131.17, "K": 146.19, "L": 131.17,
    "M": 149.21, "N": 132.12, "P": 115.13, "Q": 146.15, "R": 174.20,
    "S": 105.09, "T": 119.12, "V": 117.15, "W": 204.23, "Y": 181.19
}

# Peptide feature selection
Hydroph = set("AVILMFWY")
Small = set("AGSTCPDNV")
Tiny = set("AGSC")
Pos = set("KRH")
Neg = set("DE")
Pol = set("STNQCY")
Aromatic = set("FWYH")

###############################################################

# Initial helper functions

def build_feats(peptide: str) -> dict:
    peptide = peptide.upper().strip()
    length = len(peptide)
    if length == 0:
        return {}

    feats = {}
    feats["length"] = length

    GROUPS = {
        "hydrophobic": Hydroph,
        "polar": Pol,
        "positive": Pos,
        "negative": Neg,
        "aromatic": Aromatic,
        "small": Small,
        "tiny": Tiny,
    }

    for name, group in GROUPS.items():
        feats[f"{name}_frac"] = sum(a in group for a in peptide) / length

    feats["proline_frac"] = peptide.count("P") / length
    feats["glycine_frac"] = peptide.count("G") / length

    feats["mw_est"] = sum(AA_mol_weight.get(a, 0.0) for a in peptide)

    # amino acid composition
    for aa in AA:
        feats[f"frac_{aa}"] = peptide.count(aa) / length

    return feats

###############################################################

# Load data

df = pd.read_csv(Source_CSV)

cols = ["pdb_id", "peptide", "label", "tcren_score"]
missing = [c for c in cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df["pdb_id"] = df["pdb_id"].astype(str).str.strip().str.lower()
df["peptide"] = df["peptide"].astype(str).str.strip().str.upper()

###############################################################

# Feature table generation

feature_df = pd.DataFrame(df["peptide"].apply(build_feats).tolist())
df_out = pd.concat([df.reset_index(drop=True), feature_df], axis=1)


#print(f"Input shape: {df.shape}")
#print(f"Output shape: {df_out.shape}")
#print("\nNew columns added:")
#print(list(feature_df.columns[:15]), "...")

###############################################################

# Save
df_out.to_csv(Final_CSV, index=False)
print(f"\nSaved to: {Final_CSV.resolve()}")
