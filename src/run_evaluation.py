"""Evaluate TCR-peptide ranking models across repeated grouped splits."""

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE = Path(__file__).resolve().parent.parent

# Allow direct execution from the repository root:
# python src/run_evaluation.py
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.sequence_features import PEP_COLS, TCR_COLS


DATA_CSV = BASE / "processed" / "data_tcren_features_with_tcr.csv"
RESULTS_CSV = BASE / "results" / "repeated_eval_all_results_with_tcr.csv"
SUMMARY_CSV = BASE / "results" / "repeated_eval_summary_with_tcr.csv"

N_REPEATS = 30
TEST_SIZE = 0.30
RANDOM_SEED = 42

PEPTIDE_FEATURES = PEP_COLS
TCR_FEATURES = TCR_COLS
PEPTIDE_TCR_FEATURES = PEPTIDE_FEATURES + TCR_FEATURES
FULL_FEATURES = PEPTIDE_TCR_FEATURES + ["tcren_score"]


def make_logistic_regression(seed):
    """Create the logistic-regression pipeline used in evaluation."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )


def make_random_forest(seed):
    """Create the Random Forest used in evaluation."""
    return RandomForestClassifier(
        n_estimators=300,
        random_state=seed,
        class_weight="balanced",
    )


def ranking_metrics(test_df, score_column):
    """Calculate ranks of the cognate peptide within each held-out TCR."""
    ranks = []

    for pdb_id, group in test_df.groupby("pdb_id"):
        ranked = (
            group
            .sort_values(score_column, ascending=False)
            .reset_index(drop=True)
        )

        positive_indices = np.where(
            ranked["label"].values == 1
        )[0]

        if len(positive_indices) != 1:
            raise ValueError(
                f"{pdb_id}: expected 1 positive, "
                f"got {len(positive_indices)}"
            )

        ranks.append(int(positive_indices[0]) + 1)

    ranks = np.asarray(ranks)

    return {
        "mean_rank": ranks.mean(),
        "median_rank": np.median(ranks),
        "mrr": np.mean(1.0 / ranks),
        "top1": np.mean(ranks == 1),
        "top5": np.mean(ranks <= 5),
    }


def main():
    """Run repeated grouped evaluation and save detailed and summary results."""
    parser = argparse.ArgumentParser(
        description="Evaluate TCR-peptide ranking models across repeated grouped splits."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_CSV,
        help="Input feature CSV.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS_CSV,
        help="Detailed results CSV.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=SUMMARY_CSV,
        help="Summary results CSV.",
    )
    args = parser.parse_args()

    data_csv = args.data if args.data.is_absolute() else BASE / args.data
    results_csv = args.results if args.results.is_absolute() else BASE / args.results
    summary_csv = args.summary if args.summary.is_absolute() else BASE / args.summary

    df = pd.read_csv(data_csv)

    required_columns = (
        ["pdb_id", "peptide", "label", "tcren_score"]
        + PEPTIDE_FEATURES
        + TCR_FEATURES
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["pdb_id"] = (
        df["pdb_id"]
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

    results = []

    for repeat in range(N_REPEATS):
        split_seed = RANDOM_SEED + repeat

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=split_seed,
        )

        train_idx, test_idx = next(
            splitter.split(
                df,
                y=df["label"],
                groups=df["pdb_id"],
            )
        )

        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        train_groups = set(train_df["pdb_id"])
        test_groups = set(test_df["pdb_id"])

        overlap = train_groups & test_groups

        if overlap:
            raise RuntimeError(
                "TCR overlap detected between train and test: "
                f"{sorted(overlap)}"
            )

        y_train = train_df["label"].values
        y_test = test_df["label"].values

        feature_sets = {
            "peptide": PEPTIDE_FEATURES,
            "peptide_tcr": PEPTIDE_TCR_FEATURES,
            "full": FULL_FEATURES,
        }

        models = {}

        for feature_name, columns in feature_sets.items():
            logistic = make_logistic_regression(split_seed)
            forest = make_random_forest(split_seed)

            logistic.fit(
                train_df[columns],
                y_train,
            )

            forest.fit(
                train_df[columns],
                y_train,
            )

            models[f"logreg_{feature_name}"] = (
                logistic,
                columns,
            )

            models[f"rf_{feature_name}"] = (
                forest,
                columns,
            )

        scored = test_df.copy()

        # Lower TCRen energy is better, so negate it for ranking.
        scored["score_tcren_baseline"] = -scored["tcren_score"]

        for model_name, (model, columns) in models.items():
            scored[f"score_{model_name}"] = model.predict_proba(
                scored[columns]
            )[:, 1]

        score_columns = {
            "tcren_baseline": "score_tcren_baseline",
            **{
                model_name: f"score_{model_name}"
                for model_name in models
            },
        }

        for model_name, score_column in score_columns.items():
            scores = scored[score_column].values

            # Accuracy is retained as a supplementary classification metric.
            predictions = (scores >= 0.5).astype(int)

            row = {
                "repeat": repeat,
                "model": model_name,
                "n_train_groups": train_df["pdb_id"].nunique(),
                "n_test_groups": test_df["pdb_id"].nunique(),
                "roc_auc": roc_auc_score(y_test, scores),
                "avg_precision": average_precision_score(
                    y_test,
                    scores,
                ),
                "accuracy": accuracy_score(
                    y_test,
                    predictions,
                ),
            }

            row.update(
                ranking_metrics(
                    scored,
                    score_column,
                )
            )

            results.append(row)

    results_df = pd.DataFrame(results)

    results_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        results_csv,
        index=False,
    )

    metrics = [
        "roc_auc",
        "avg_precision",
        "accuracy",
        "mean_rank",
        "median_rank",
        "mrr",
        "top1",
        "top5",
    ]

    summary_rows = []

    for model_name, group in results_df.groupby("model"):
        row = {"model": model_name}

        for metric in metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std()

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        summary_csv,
        index=False,
    )

    print(
        summary_df
        .sort_values("mrr_mean", ascending=False)
        .to_string(index=False)
    )

    print(
        f"\nSaved detailed results to: "
        f"{results_csv.resolve()}"
    )

    print(
        f"Saved summary to: "
        f"{summary_csv.resolve()}"
    )


if __name__ == "__main__":
    main()
