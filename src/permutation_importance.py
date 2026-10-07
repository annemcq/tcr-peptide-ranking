"""Estimate feature importance on held-out TCR systems using permutation importance."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit


BASE = Path(__file__).resolve().parent.parent
DATA_CSV = BASE / "processed" / "data_tcren_features_with_tcr.csv"

RESULTS_DIR = BASE / "results"
OUTPUT_CSV = RESULTS_DIR / "permutation_importance.csv"
PERFORMANCE_OUTPUT_CSV = RESULTS_DIR / "permutation_importance_performance.csv"

N_SPLITS = 30
TEST_SIZE = 0.30
RANDOM_SEED = 42


PEPTIDE_FEATURES = [
    "length",
    "hydrophobic_frac",
    "polar_frac",
    "positive_frac",
    "negative_frac",
    "aromatic_frac",
    "small_frac",
    "tiny_frac",
    "proline_frac",
    "glycine_frac",
    "mw_est",
    "frac_A",
    "frac_C",
    "frac_D",
    "frac_E",
    "frac_F",
    "frac_G",
    "frac_H",
    "frac_I",
    "frac_K",
    "frac_L",
    "frac_M",
    "frac_N",
    "frac_P",
    "frac_Q",
    "frac_R",
    "frac_S",
    "frac_T",
    "frac_V",
    "frac_W",
    "frac_Y",
]


TCR_FEATURES = [
    "cdr3a_len",
    "cdr3b_len",
    "cdr3a_hydrophobic_frac",
    "cdr3a_positive_frac",
    "cdr3a_negative_frac",
    "cdr3a_aromatic_frac",
    "cdr3a_glycine_frac",
    "cdr3a_proline_frac",
    "cdr3b_hydrophobic_frac",
    "cdr3b_positive_frac",
    "cdr3b_negative_frac",
    "cdr3b_aromatic_frac",
    "cdr3b_glycine_frac",
    "cdr3b_proline_frac",
]


FEATURE_COLS = PEPTIDE_FEATURES + TCR_FEATURES + ["tcren_score"]


def main():
    """Compute permutation importance across repeated grouped splits."""

    df = pd.read_csv(DATA_CSV)

    required_cols = FEATURE_COLS + ["label", "pdb_id"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[FEATURE_COLS]
    y = df["label"]
    groups = df["pdb_id"]

    all_importances = []
    split_performance = []

    for i in range(N_SPLITS):
        split_seed = RANDOM_SEED + i

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

        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])

        if train_groups & test_groups:
            raise RuntimeError(
                "TCR overlap detected between train and test."
            )

        model = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=split_seed,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        predictions = model.predict_proba(X_test)[:, 1]

        ap = average_precision_score(
            y_test,
            predictions,
        )

        split_performance.append(
            {
                "split": i,
                "seed": split_seed,
                "average_precision": ap,
            }
        )

        result = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="average_precision",
            n_repeats=20,
            random_state=split_seed,
            n_jobs=-1,
        )

        for feature, importance in zip(
            FEATURE_COLS,
            result.importances_mean,
        ):
            all_importances.append(
                {
                    "split": i,
                    "seed": split_seed,
                    "feature": feature,
                    "importance": importance,
                }
            )

        print(
            f"Split {i + 1:02d}/{N_SPLITS} "
            f"(seed {split_seed}) "
            f"- held-out AP: {ap:.3f}"
        )

    importance_df = pd.DataFrame(all_importances)

    summary = (
        importance_df
        .groupby("feature")["importance"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    performance_df = pd.DataFrame(split_performance)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    performance_df.to_csv(
        PERFORMANCE_OUTPUT_CSV,
        index=False,
    )

    print("\nHeld-out performance:")
    print(
        f"Mean AP: "
        f"{performance_df['average_precision'].mean():.3f}"
    )
    print(
        f"SD AP:   "
        f"{performance_df['average_precision'].std():.3f}"
    )

    print("\nPermutation importance:")
    print(
        summary.to_string(index=False)
    )

    print(
        f"\nSaved feature importance to: "
        f"{OUTPUT_CSV.resolve()}"
    )

    print(
        f"Saved split performance to: "
        f"{PERFORMANCE_OUTPUT_CSV.resolve()}"
    )


if __name__ == "__main__":
    main()
