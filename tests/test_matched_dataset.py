"""Tests for composition-matched negative generation."""

from collections import Counter

import pytest

from src.build_matched_dataset import generate_matched_negatives


def test_matched_negatives_have_same_length():
    positive = "LLFGYPVYV"
    negatives = generate_matched_negatives(positive, n_neg=100, seed=42)
    assert all(len(peptide) == len(positive) for peptide in negatives)


def test_matched_negatives_have_same_composition():
    positive = "LLFGYPVYV"
    negatives = generate_matched_negatives(positive, n_neg=100, seed=42)
    expected = Counter(positive)
    assert all(Counter(peptide) == expected for peptide in negatives)


def test_matched_negatives_are_unique():
    positive = "LLFGYPVYV"
    negatives = generate_matched_negatives(positive, n_neg=100, seed=42)
    assert len(negatives) == len(set(negatives))


def test_matched_negatives_exclude_positive():
    positive = "LLFGYPVYV"
    negatives = generate_matched_negatives(positive, n_neg=100, seed=42)
    assert positive not in negatives


def test_impossible_number_of_negatives_raises():
    positive = "AAAA"
    with pytest.raises(ValueError):
        generate_matched_negatives(positive, n_neg=100, seed=42)
