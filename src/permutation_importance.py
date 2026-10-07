"""Estimate feature importance on held-out TCR systems using permutation importance."""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit


BASE = Path(__file__).resolve().parent.parent
DATA_CSV = BASE / "processed" / "data_tcren_features_with_tcr.csv"

RESULTS_DIR = BASE / "results"
OUTPUT_CSV = RESULTS_DIR / "permutation_importance.csv"

N_SPLITS = 30
TEST_SIZE = 0.30
RANDOM_SEED = 42


def main():
    """Compute permutation importance across repeated grouped splits."""

    df = pd.read_csv(DATA_CSV)

    feature_cols = [
        col
        for col in df.columns
        if col.startswith("pep_")
        or col.startswith("tcr_")
        or col == "tcren_score"
    ]

    if not feature_cols:
        raise ValueError("No model features were found.")

    required_cols = ["label", "pdb_id"]
    missing = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    X = df[feature_cols]
    y = df["label"]
    groups = df["pdb_id"]

    all_importances = []

    for split_seed in range(N_SPLITS):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=split_seed,
        )

        train_idx, test_idx = next(
            splitter.split(X, y, groups=groups)
        )

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        train_groups = set(
            groups.iloc[train_idx]
        )
        test_groups = set(
            groups.iloc[test_idx]
        )

        if train_groups & test_groups:
            raise RuntimeError(
                "TCR overlap detected between train and test."
            )

        model = RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        result = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="average_precision",
            n_repeats=20,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )

        for feature, importance in zip(
            feature_cols,
            result.importances_mean,
        ):
            all_importances.append(
                {
                    "split": split_seed,
                    "feature": feature,
                    "importance": importance,
                }
            )

        predictions = model.predict_proba(X_test)[:, 1]

        ap = average_precision_score(
            y_test,
            predictions,
        )

        print(
            f"Split {split_seed + 1:02d}/{N_SPLITS} "
            f"- held-out AP: {ap:.3f}"
        )

    importance_df = pd.DataFrame(
        all_importances
    )

    summary = (
        importance_df
        .groupby("feature")["importance"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values(
            "mean",
            ascending=False,
        )
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print("\nPermutation importance:")
    print(summary.to_string(index=False))

    print(
        f"\nSaved results to: "
        f"{OUTPUT_CSV.resolve()}"
    )


if __name__ == "__main__":
    main()
