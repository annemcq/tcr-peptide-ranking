"""Tests for ranking metrics."""

import pandas as pd

from src.run_evaluation import ranking_metrics


def test_ranking_metrics_average_ties():
    df = pd.DataFrame(
        {
            "pdb_id": ["1", "1", "1", "2", "2"],
            "label": [1, 0, 0, 0, 1],
            "score": [0.5, 0.5, 0.5, 0.9, 0.1],
        }
    )

    metrics = ranking_metrics(df, "score")

    # In TCR 1 all three candidates tie, so the positive gets rank 2
    # rather than an arbitrary rank determined by row order.
    assert metrics["mean_rank"] == 1.5
    assert metrics["top1"] == 0.5
    assert metrics["top5"] == 1.0
    assert metrics["mrr"] == (0.5 + 1.0) / 2


def test_ranking_metrics_requires_one_positive_per_group():
    df = pd.DataFrame(
        {
            "pdb_id": ["1", "1"],
            "label": [1, 1],
            "score": [0.5, 0.4],
        }
    )

    try:
        ranking_metrics(df, "score")
    except ValueError as exc:
        assert "expected 1 positive" in str(exc)
    else:
        raise AssertionError("Expected ValueError for multiple positives.")
