"""Add peptide sequence features to the TCR-peptide dataset."""

from pathlib import Path

import pandas as pd

from src.sequence_features import build_peptide_features


BASE = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    BASE
    / "processed"
    / "data_tcren.csv"
)

OUTPUT_CSV = (
    BASE
    / "processed"
    / "data_tcren_features.csv"
)


def main():
    """Calculate peptide features for every TCR-peptide pair."""

    df = pd.read_csv(INPUT_CSV)

    required_cols = ["peptide"]

    missing = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    missing_peptides = df["peptide"].isna()

    if missing_peptides.any():
        raise ValueError(
            f"Found {missing_peptides.sum()} rows "
            "with missing peptide sequences."
        )

    df["peptide"] = (
        df["peptide"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    empty_peptides = df["peptide"] == ""

    if empty_peptides.any():
        raise ValueError(
            f"Found {empty_peptides.sum()} rows "
            "with empty peptide sequences."
        )

    feature_rows = [
        build_peptide_features(peptide)
        for peptide in df["peptide"]
    ]

    peptide_features = pd.DataFrame(
        feature_rows,
        index=df.index,
    )

    df_out = pd.concat(
        [
            df,
            peptide_features,
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
        f"Added peptide features to "
        f"{len(df_out)} TCR-peptide pairs."
    )

    print(
        f"Saved dataset to: "
        f"{OUTPUT_CSV.resolve()}"
    )


if __name__ == "__main__":
    main()
