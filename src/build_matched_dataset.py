"""Build a composition-matched TCR-peptide ranking dataset."""

from pathlib import Path
import random
from collections import Counter
import math

import pandas as pd


BASE = Path(__file__).resolve().parent.parent
SOURCE_CSV = BASE / "processed" / "first_dataset_pairs.csv"
OUTPUT_CSV = BASE / "processed" / "matched_composition_pairs.csv"

N_NEG = 100
RANDOM_SEED = 42


def _count_unique_permutations(sequence):
    """Return the number of distinct permutations of a sequence."""
    counts = Counter(sequence)
    total = math.factorial(len(sequence))
    for count in counts.values():
        total //= math.factorial(count)
    return total


def generate_matched_negatives(positive_peptide, n_neg=N_NEG, seed=RANDOM_SEED):
    """Generate unique negatives preserving amino-acid composition."""
    positive = positive_peptide.strip().upper()
    max_unique = _count_unique_permutations(positive) - 1

    if max_unique < n_neg:
        raise ValueError(
            f"Cannot generate {n_neg} unique composition-matched negatives "
            f"for peptide '{positive}'. Only {max_unique} alternatives are possible."
        )

    rng = random.Random(seed)
    residues = list(positive)
    negatives = set()

    while len(negatives) < n_neg:
        shuffled = residues.copy()
        rng.shuffle(shuffled)
        candidate = "".join(shuffled)
        if candidate != positive:
            negatives.add(candidate)

    return sorted(negatives)


def main():
    source = pd.read_csv(SOURCE_CSV)

    required_columns = ["pdb_id", "peptide", "label"]
    missing = [column for column in required_columns if column not in source.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    source["pdb_id"] = source["pdb_id"].astype(str).str.strip().str.lower()
    source["peptide"] = source["peptide"].astype(str).str.strip().str.upper()

    rows = []

    for pdb_id, group in source.groupby("pdb_id"):
        positives = group[group["label"] == 1]

        if len(positives) != 1:
            raise ValueError(
                f"{pdb_id}: expected exactly one positive peptide, found {len(positives)}."
            )

        positive_peptide = positives.iloc[0]["peptide"]
        rows.append({"pdb_id": pdb_id, "peptide": positive_peptide, "label": 1})

        negatives = generate_matched_negatives(
            positive_peptide,
            n_neg=N_NEG,
            seed=RANDOM_SEED + hash(pdb_id) % 100000,
        )

        for peptide in negatives:
            rows.append({"pdb_id": pdb_id, "peptide": peptide, "label": 0})

    dataset = pd.DataFrame(rows)
    expected_rows = source["pdb_id"].nunique() * (N_NEG + 1)

    if len(dataset) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, but generated {len(dataset)}.")

    for pdb_id, group in dataset.groupby("pdb_id"):
        positive = group.loc[group["label"] == 1, "peptide"].iloc[0]
        positive_counts = Counter(positive)

        for peptide in group.loc[group["label"] == 0, "peptide"]:
            if Counter(peptide) != positive_counts:
                raise RuntimeError(f"Composition mismatch for {pdb_id}: {peptide} vs {positive}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_CSV, index=False)

    print(
        f"Built composition-matched dataset with "
        f"{dataset['pdb_id'].nunique()} TCR systems and {len(dataset)} total pairs."
    )
    print(f"Saved: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
