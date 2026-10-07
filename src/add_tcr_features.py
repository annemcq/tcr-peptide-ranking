"""Add TCR sequence features to the TCR-peptide dataset."""

from pathlib import Path
import sys

import pandas as pd


BASE = Path(__file__).resolve().parent.parent

# Allow this script to be run directly from the repository root:
# python src/add_tcr_features.py
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.sequence_features import build_tcr_features


INPUT_CSV = (
    BASE
    / "processed"
    / "data_tcren_features.csv"
)

OUTPUT_CSV = (
    BASE
    / "processed"
    / "data_tcren_features_with_tcr.csv"
)


def main():
    """Add CDR3 alpha/beta sequence features to every TCR-peptide pair."""

    df = pd.read_csv(INPUT_CSV)

    required_columns = ["cdr3a", "cdr3b"]
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    missing_sequences = (
        df["cdr3a"].isna()
        | df["cdr3b"].isna()
    )

    if missing_sequences.any():
        raise ValueError(
            f"Found {missing_sequences.sum()} rows "
            "with missing CDR3 sequences."
        )

    df["cdr3a"] = (
        df["cdr3a"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["cdr3b"] = (
        df["cdr3b"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    empty_sequences = (
        (df["cdr3a"] == "")
        | (df["cdr3b"] == "")
    )

    if empty_sequences.any():
        raise ValueError(
            f"Found {empty_sequences.sum()} rows "
            "with empty CDR3 sequences."
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

    # Avoid duplicate columns if the script is rerun on an
    # already processed dataset.
    duplicate_columns = [
        column
        for column in tcr_features.columns
        if column in df.columns
    ]

    if duplicate_columns:
        df = df.drop(
            columns=duplicate_columns
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
