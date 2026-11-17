"""Tests for reproducible multi-draw negative-set generation and aggregation."""
from collections import Counter

import numpy as np
import pandas as pd

from src.evaluate_matched_multidraw import make_draw, summarise


def test_draw_is_reproducible_and_preserves_composition():
    positives = pd.DataFrame([
        {"pdb_id": "1abc", "peptide": "ACDEFGHIK", "cdr3a": "ACDE", "cdr3b": "GHI"},
    ])
    contacts = {"1abc": [(0, "A"), (1, "C")]}
    potential = {(a, b): float(i + j) for i, a in enumerate("AC") for j, b in enumerate("ACDEFGHIK")}
    first = make_draw(positives, contacts, potential, draw=0, base_seed=10000)
    same = make_draw(positives, contacts, potential, draw=0, base_seed=10000)
    other = make_draw(positives, contacts, potential, draw=1, base_seed=10000)
    assert first.equals(same)
    assert len(first) == 101
    assert first.label.sum() == 1
    assert first.peptide.nunique() == 101
    assert all(Counter(p) == Counter("ACDEFGHIK") for p in first.peptide)
    assert first.peptide.tolist() != other.peptide.tolist()


def test_summary_uses_draw_means_not_all_split_rows():
    split_rows = pd.DataFrame([
        {"draw": 0, "model": "rf_full", "mrr": 0.1, "top1": 0., "top5": 0., "roc_auc": 0.5, "avg_precision": 0.1},
        {"draw": 0, "model": "rf_full", "mrr": 0.3, "top1": 0., "top5": 0., "roc_auc": 0.5, "avg_precision": 0.1},
        {"draw": 1, "model": "rf_full", "mrr": 0.6, "top1": 1., "top5": 1., "roc_auc": 0.9, "avg_precision": 0.5},
        {"draw": 1, "model": "rf_full", "mrr": 0.8, "top1": 1., "top5": 1., "roc_auc": 0.9, "avg_precision": 0.5},
    ])
    draw, summary = summarise(split_rows)
    assert np.allclose(draw.mrr, [0.2, 0.7])
    assert np.isclose(summary["mrr_mean_across_draws"].iloc[0], 0.45)
    assert np.isclose(summary["mrr_std_across_draws"].iloc[0], np.std([0.2, 0.7], ddof=1))
