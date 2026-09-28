# DATASET CONSTRUCTION

# Import necessary libraries
import pandas as pd
import random
from pathlib import Path

# Baseline settings for code
BASE = Path(__file__).resolve().parent.parent
Source_CSV = BASE / "data" / "summary_PDB_structures.csv"
Final_CSV = BASE / "processed" / "first_dataset_pairs.csv"
Contact_map_CSV = BASE / "data" / "contact_maps_PDB.csv"

N_TCRS = 25 # range between 20-30 TCRs
N_NEG = 100 # number of negative peptides per TCR
R_S = 42

AA = list("ACDEFGHIKLMNPQRSTVWY") # aminoacids

random.seed(R_S) # initialize seed

###############################################################

# Initial helper functions

def make_random_peptide(n):
    # generate a random peptide string of length n
    return "".join(random.choice(AA) for _ in range(n))


def generate_negatives(pos_pep, n_neg):
    # generates negative peptides of equal length to each given positive peptide
    length = len(pos_pep)
    negs = set()

    while len(negs) < n_neg:
        candidate = make_random_peptide(length)
        if candidate != pos_pep:
            negs.add(candidate)

    return list(negs)

###############################################################

# Inspection of source csv
df = pd.read_csv(Source_CSV)
df_contacts = pd.read_csv(Contact_map_CSV)

print("Columns:", list(df.columns), "\n")

# filter non-redundant entries (if column exists)
if "nonred" in df.columns:
    mask = df["nonred"].astype(str).str.lower().isin(["true", "1", "yes"])
    df = df[mask]

for col in ["pdb.id", "peptide"]:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not present in source data")

# drop incomplete rows
df = df.dropna(subset=["pdb.id", "peptide"]).copy()

# clean formatting
df["pdb.id"] = df["pdb.id"].str.strip().str.lower()
df["peptide"] = df["peptide"].str.strip().str.upper()
# remove duplicates (just in case)
df = df.drop_duplicates("pdb.id")

print(f"Final systems: {len(df)}")

# Filter for Pdb_ids that have contact maps
available_contacts = set(df_contacts["pdb.id"].unique())
df = df[df["pdb.id"].isin(available_contacts)].copy()

print("Usable systems after contact-map filtering:", len(df))

# Test a subset for reproducibility
df_subset = df.iloc[:N_TCRS].copy()

print(f"Using {len(df_subset)} systems")
print(df_subset[["pdb.id", "peptide"]].head(), "\n")


###############################################################
# Construction of dataset: positives + negatives

rows = []

for _, row in df_subset.iterrows():
    pdb_id = row["pdb.id"]
    pos_pep = row["peptide"]

    # Positive row
    rows.append({
        "pdb_id": pdb_id,
        "peptide": pos_pep,
        "label": 1
    })

    # Negative rows
    negatives = generate_negatives(
        pos_pep=pos_pep,
        n_neg=N_NEG
    )

    for neg_peptide in negatives:
        rows.append({
            "pdb_id": pdb_id,
            "peptide": neg_peptide,
            "label": 0
        })

dataset = pd.DataFrame(rows)

print("Dataset shape:", dataset.shape)
print(dataset.head(10))
print()
print("Label counts:")
print(dataset["label"].value_counts())

###############################################################
# Save
dataset.to_csv(Final_CSV, index=False)
print(f"\nSaved dataset to: {Final_CSV.resolve()}")