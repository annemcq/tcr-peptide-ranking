"""Build the TCR-peptide ranking dataset."""

from pathlib import Path
import random

import pandas as pd


BASE = Path(__file__).resolve().parent.parent

SOURCE_CSV = BASE / "data" / "summary_PDB_structures.csv"
CONTACT_MAP_CSV = BASE / "data" / "contact_maps_PDB.csv"
FINAL_CSV = BASE / "processed" / "first_dataset_pairs.csv"

N_TCRS = 25
N_NEG = 100
RANDOM_SEED = 42

AA = list("ACDEFGHIKLMNPQRSTVWY")


def make_random_peptide(length):
    """Generate a random amino-acid sequence of a given length."""
    return "".join(random.choice(AA) for _ in range(length))


def generate_negatives(pos_pep, n_neg):
    """Generate unique random negatives with the same length as the positive."""
    negatives = set()

    while len(negatives) < n_neg:
        candidate = make_random_peptide(len(pos_pep))

        if candidate != pos_pep:
            negatives.add(candidate)

    return sorted(negatives)


def main():
    """Construct positive and negative TCR-peptide pairs."""

    random.seed(RANDOM_SEED)

    source = pd.read_csv(SOURCE_CSV)
    contacts = pd.read_csv(CONTACT_MAP_CSV)

    required_source_cols = ["pdb.id", "peptide"]
    required_contact_cols = ["pdb.id"]

    missing_source = [
        col for col in required_source_cols
        if col not in source.columns
    ]

    missing_contacts = [
        col for col in required_contact_cols
        if col not in contacts.columns
    ]

    if missing_source:
        raise ValueError(
            f"Missing required source columns: {missing_source}"
        )

    if missing_contacts:
        raise ValueError(
            f"Missing required contact-map columns: {missing_contacts}"
        )

    source["pdb.id"] = (
        source["pdb.id"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    contacts["pdb.id"] = (
        contacts["pdb.id"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if "nonred" in source.columns:
        source = source[
            source["nonred"] == 1
        ].copy()

    source = source.dropna(
        subset=["pdb.id", "peptide"]
    ).copy()

    source["peptide"] = (
        source["peptide"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    source = source.drop_duplicates(
        subset=["pdb.id"]
    ).copy()

    pdbs_with_contacts = set(
        contacts["pdb.id"]
        .dropna()
        .unique()
    )

    source = source[
        source["pdb.id"].isin(pdbs_with_contacts)
    ].copy()

    source = source.head(N_TCRS)

    if len(source) < N_TCRS:
        raise ValueError(
            f"Only {len(source)} eligible TCR systems were found; "
            f"{N_TCRS} are required."
        )

    rows = []

    for _, row in source.iterrows():
        pdb_id = row["pdb.id"]
        positive_peptide = row["peptide"]

        rows.append(
            {
                "pdb_id": pdb_id,
                "peptide": positive_peptide,
                "label": 1,
            }
        )

        negatives = generate_negatives(
            positive_peptide,
            N_NEG,
        )

        for peptide in negatives:
            rows.append(
                {
                    "pdb_id": pdb_id,
                    "peptide": peptide,
                    "label": 0,
                }
            )

    dataset = pd.DataFrame(rows)

    expected_rows = len(source) * (N_NEG + 1)

    if len(dataset) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rows, "
            f"but generated {len(dataset)}."
        )

    FINAL_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_csv(
        FINAL_CSV,
        index=False,
    )

    print(
        f"Built dataset with {len(source)} TCR systems "
        f"and {len(dataset)} total pairs."
    )

    print(
        f"Saved dataset to: {FINAL_CSV.resolve()}"
    )


if __name__ == "__main__":
    main()
