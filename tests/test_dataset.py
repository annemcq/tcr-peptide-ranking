"""Tests for TCR-peptide dataset construction."""

import random

from src.build_dataset import generate_negatives


def test_generate_negatives_returns_requested_number():
    """The requested number of negative peptides should be generated."""
    random.seed(42)

    positive = "LLFGYPVYV"
    negatives = generate_negatives(positive, n_neg=100)

    assert len(negatives) == 100


def test_generate_negatives_are_unique():
    """Generated negative peptides should not contain duplicates."""
    random.seed(42)

    positive = "LLFGYPVYV"
    negatives = generate_negatives(positive, n_neg=100)

    assert len(negatives) == len(set(negatives))


def test_generate_negatives_match_positive_length():
    """Negative peptides should have the same length as the positive."""
    random.seed(42)

    positive = "LLFGYPVYV"
    negatives = generate_negatives(positive, n_neg=100)

    assert all(
        len(peptide) == len(positive)
        for peptide in negatives
    )


def test_generate_negatives_exclude_positive():
    """The cognate peptide should never appear among the negatives."""
    random.seed(42)

    positive = "LLFGYPVYV"
    negatives = generate_negatives(positive, n_neg=100)

    assert positive not in negatives
