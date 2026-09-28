"""
Tests for src/sequence_features.py (the reusable feature-building functions
used by the inference showcase notebook) and the Holm-Bonferroni correction
used by src/paired_significance_test.py.
"""

import numpy as np
import pytest

from src.sequence_features import (
    build_peptide_features,
    build_tcr_features,
    build_feature_row,
    PEP_COLS,
    TCR_COLS,
    PEP_TCR_COLS,
)
from src.paired_significance_test import holm_bonferroni


# ------------------------------------------------------------------
# Peptide features
# ------------------------------------------------------------------

def test_peptide_features_has_expected_keys():
    feats = build_peptide_features("LLFGYPVYV")
    assert set(feats.keys()) == set(PEP_COLS)


def test_peptide_feature_fractions_sum_reasonably():
    feats = build_peptide_features("LLFGYPVYV")
    assert feats["length"] == 9
    # every amino-acid fraction should be between 0 and 1, and sum to 1 across the 20 aa's
    aa_fracs = [feats[f"frac_{aa}"] for aa in "ACDEFGHIKLMNPQRSTVWY"]
    assert all(0.0 <= f <= 1.0 for f in aa_fracs)
    assert abs(sum(aa_fracs) - 1.0) < 1e-9


def test_peptide_features_rejects_empty_sequence():
    with pytest.raises(ValueError):
        build_peptide_features("")


def test_peptide_features_case_insensitive():
    upper = build_peptide_features("LLFGYPVYV")
    lower = build_peptide_features("llfgypvyv")
    assert upper == lower


# ------------------------------------------------------------------
# TCR (CDR3a/CDR3b) features
# ------------------------------------------------------------------

def test_tcr_features_has_expected_keys():
    feats = build_tcr_features("CAVTTDSWGKLQF", "CASRPGLAGGRPEQYF")
    assert set(feats.keys()) == set(TCR_COLS)


def test_tcr_features_lengths_match_input():
    feats = build_tcr_features("CAVTTDSWGKLQF", "CASRPGLAGGRPEQYF")
    assert feats["cdr3a_len"] == len("CAVTTDSWGKLQF")
    assert feats["cdr3b_len"] == len("CASRPGLAGGRPEQYF")


# ------------------------------------------------------------------
# Combined feature row (used directly by the inference notebook)
# ------------------------------------------------------------------

def test_feature_row_matches_column_order():
    row = build_feature_row("LLFGYPVYV", "CAVTTDSWGKLQF", "CASRPGLAGGRPEQYF")
    assert list(row.keys()) == PEP_TCR_COLS
    assert len(row) == len(PEP_COLS) + len(TCR_COLS)


# ------------------------------------------------------------------
# Holm-Bonferroni correction (used by the paired significance test)
# ------------------------------------------------------------------

def test_holm_bonferroni_preserves_order_of_significance():
    # a clearly-significant p-value should remain smaller after correction
    # than a clearly non-significant one
    pvalues = np.array([0.001, 0.5, 0.02])
    adjusted = holm_bonferroni(pvalues)
    assert adjusted[0] < adjusted[1]
    assert adjusted[2] < adjusted[1]


def test_holm_bonferroni_never_decreases_pvalues():
    pvalues = np.array([0.01, 0.02, 0.03, 0.04])
    adjusted = holm_bonferroni(pvalues)
    assert all(adjusted >= pvalues - 1e-12)


def test_holm_bonferroni_caps_at_one():
    pvalues = np.array([0.9, 0.95, 0.99])
    adjusted = holm_bonferroni(pvalues)
    assert all(adjusted <= 1.0)
