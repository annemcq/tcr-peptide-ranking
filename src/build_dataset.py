"""Construct the TCR-peptide dataset used in the ranking experiments."""

import random
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parent.parent
SOURCE_CSV = BASE / "data" / "summary_PDB_structures.csv"
FINAL_CSV = BASE / "processed" / "first_dataset_pairs.csv"
CONTACT_MAP_CSV = BASE / "data" / "contact_maps_PDB.csv"

N_TCRS = 25
N_NEG = 100
RANDOM_SEED = 42

AA = list("ACDEFGHIKLMNPQRSTVWY")


def make_random_peptide(length):
    """Generate a random amino-acid sequence of a given length."""
    return "".join(random.choice(AA) for _ in range(length))


def generate_negatives(pos_pep, n_neg):
    """Generate unique random peptides with the same length as the positive."""
    length = len(pos_pep)
    negatives = set()

    while len(negatives) < n_neg:
        candidate = make_random_peptide(length)

        if candidate != pos_pep:
            negatives.add(candidate)

    return list(negatives)


def main():
    """Build and save the positive/negative TCR-peptide dataset."""

    # Keep dataset generation reproducible.
    random.seed(RANDOM_SEED)

    df = pd.read_csv(SOURCE_CSV)
    df_contacts = pd.read_csv(CONTACT_MAP_CSV)

    print("Columns:", list(df.columns), "\n")

    # Keep only non-redundant systems when this annotation is available.
    if "nonred" in df.columns:
        mask = (
            df["nonred"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )
        df = df[mask]

    # Check that the columns required to construct the dataset are present.
    for col in ["pdb.id", "peptide"]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not present in source data"
            )

    # Remove incomplete entries and standardize formatting.
    df = df.dropna(subset=["pdb.id", "peptide"]).copy()

    df["pdb.id"] = (
        df["pdb.id"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["peptide"] = (
        df["peptide"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Keep one peptide per PDB system.
    df = df.drop_duplicates("pdb.id")

    print(f"Final systems: {len(df)}")

    # Only retain systems for which a contact map is available.
    available_contacts = set(
        df_contacts["pdb.id"]
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
    )

    df = df[df["pdb.id"].isin(available_contacts)].copy()

    print(
        "Usable systems after contact-map filtering:",
        len(df),
    )

    # Use the first N_TCRS systems for the experiment.
    df_subset = df.iloc[:N_TCRS].copy()

    print(f"Using {len(df_subset)} systems")
    print(df_subset[["pdb.id", "peptide"]].head(), "\n")

    # Construct one positive and N_NEG negative peptides per TCR.
    rows = []

    for _, row in df_subset.iterrows():
        pdb_id = row["pdb.id"]
        pos_pep = row["peptide"]

        rows.append(
            {
                "pdb_id": pdb_id,
                "peptide": pos_pep,
                "label": 1,
            }
        )

        negatives = generate_negatives(
            pos_pep=pos_pep,
            n_neg=N_NEG,
        )

        for neg_peptide in negatives:
            rows.append(
                {
                    "pdb_id": pdb_id,
                    "peptide": neg_peptide,
                    "label": 0,
                }
            )

    dataset = pd.DataFrame(rows)

    print("Dataset shape:", dataset.shape)
    print(dataset.head(10))
    print()
    print("Label counts:")
    print(dataset["label"].value_counts())

    dataset.to_csv(FINAL_CSV, index=False)

    print(f"\nSaved dataset to: {FINAL_CSV.resolve()}")


if __name__ == "__main__":
    main()
