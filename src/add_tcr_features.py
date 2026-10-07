"""Add CDR3 sequence features to the TCR-peptide dataset."""

from pathlib import Path

import pandas as pd

from src.sequence_features import build_tcr_features


BASE = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    BASE
    / "processed"
    / "data_tcren_features.csv"
)

TCR_SOURCE_CSV = (
    BASE
    / "data"
    / "summary_PDB_structures.csv"
)

OUTPUT_CSV = (
    BASE
    / "processed"
    / "data_tcren_features_with_tcr.csv"
)


def main():
    """Merge CDR3 sequences and calculate TCR sequence features."""

    df = pd.read_csv(INPUT_CSV)
    tcr_source = pd.read_csv(TCR_SOURCE_CSV)

    required_dataset_cols = ["pdb_id"]
    required_tcr_cols = [
        "pdb.id",
        "cdr3a",
        "cdr3b",
    ]

    missing_dataset_cols = [
        col
        for col in required_dataset_cols
        if col not in df.columns
    ]

    missing_tcr_cols = [
        col
        for col in required_tcr_cols
        if col not in tcr_source.columns
    ]

    if missing_dataset_cols:
        raise ValueError(
            "Missing required dataset columns: "
            f"{missing_dataset_cols}"
        )

    if missing_tcr_cols:
        raise ValueError(
            "Missing required TCR sequence columns: "
            f"{missing_tcr_cols}"
        )

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
            "Missing CDR3 sequences for "
            f"PDB IDs: {missing_pdbs}"
        )

    for col in ["cdr3a", "cdr3b"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        empty_sequences = df[col] == ""

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
                f"Empty {col} sequences for "
                f"PDB IDs: {affected_pdbs}"
            )

    tcr_feature_rows = [
        build_tcr_features(cdr3a, cdr3b)
        for cdr3a, cdr3b in zip(
            df["cdr3a"],
            df["cdr3b"],
        )
    ]

    tcr_features = pd.DataFrame(
        tcr_feature_rows,
        index=df.index,
    )

    df = pd.concat(
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

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print(
        f"Added TCR features to "
        f"{len(df)} TCR-peptide pairs."
    )

    print(
        f"Saved dataset to: "
        f"{OUTPUT_CSV.resolve()}"
    )


if __name__ == "__main__":
    main()
