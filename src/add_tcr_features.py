"""Add TCR sequence features to the TCR-peptide dataset."""

from pathlib import Path
import sys

import pandas as pd


BASE = Path(__file__).resolve().parent.parent

# Allow direct execution from the repository root:
# python src/add_tcr_features.py
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.sequence_features import build_tcr_features


INPUT_CSV = BASE / "processed" / "data_tcren_features.csv"
TCR_SOURCE_CSV = BASE / "data" / "summary_PDB_structures.csv"
OUTPUT_CSV = BASE / "processed" / "data_tcren_features_with_tcr.csv"


def main():
    """Merge CDR3 sequences and calculate TCR sequence features."""

    df = pd.read_csv(INPUT_CSV)
    tcr_source = pd.read_csv(TCR_SOURCE_CSV)

    if "pdb_id" not in df.columns:
        raise ValueError(
            "Missing required column in input dataset: pdb_id"
        )

    required_tcr_columns = [
        "pdb.id",
        "cdr3a",
        "cdr3b",
    ]

    missing_tcr_columns = [
        column
        for column in required_tcr_columns
        if column not in tcr_source.columns
    ]

    if missing_tcr_columns:
        raise ValueError(
            "Missing required columns in TCR source: "
            f"{missing_tcr_columns}"
        )

    # Normalize PDB identifiers before merging.
    df["pdb_id"] = (
        df["pdb_id"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    tcr_source["pdb.id"] = (
        tcr_source["pdb.id"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Keep one CDR3 alpha/beta pair for each PDB structure.
    tcr_sequences = (
        tcr_source[
            ["pdb.id", "cdr3a", "cdr3b"]
        ]
        .rename(
            columns={
                "pdb.id": "pdb_id",
            }
        )
        .drop_duplicates(
            subset=["pdb_id"]
        )
        .copy()
    )

    df = df.merge(
        tcr_sequences,
        on="pdb_id",
        how="left",
        validate="many_to_one",
    )

    # Check missing values before converting sequences to strings.
    missing_sequences = (
        df[["cdr3a", "cdr3b"]]
        .isna()
        .any(axis=1)
    )

    if missing_sequences.any():
        missing_pdbs = sorted(
            df.loc[
                missing_sequences,
                "pdb_id",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Missing CDR3 sequences for PDB IDs: "
            f"{missing_pdbs}"
        )

    for column in ["cdr3a", "cdr3b"]:
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        empty_sequences = df[column] == ""

        if empty_sequences.any():
            affected_pdbs = sorted(
                df.loc[
                    empty_sequences,
                    "pdb_id",
                ]
                .unique()
                .tolist()
            )

            raise ValueError(
                f"Empty {column} sequences for PDB IDs: "
                f"{affected_pdbs}"
            )

    feature_rows = [
        build_tcr_features(cdr3a, cdr3b)
        for cdr3a, cdr3b in zip(
            df["cdr3a"],
            df["cdr3b"],
        )
    ]

    tcr_features = pd.DataFrame(
        feature_rows,
        index=df.index,
    )

    df_out = pd.concat(
        [
            df,
            tcr_features,
        ],
        axis=1,
    )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df_out.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print(
        f"Added TCR features to "
        f"{len(df_out)} TCR-peptide pairs."
    )

    print(
        f"Saved dataset to: "
        f"{OUTPUT_CSV.resolve()}"
    )


if __name__ == "__main__":
    main()
