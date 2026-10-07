"""Tests for TCRen scoring."""

import pandas as pd
import pytest

from src.compute_tcren_scores import compute_tcren_score


def test_compute_tcren_score_sums_contact_potentials():
    """The score should equal the sum of the relevant TCRen potentials."""

    contacts = pd.DataFrame(
        {
            "pdb.id": ["test", "test"],
            "residue.index.to": [0, 1],
            "residue.aa.from": ["A", "G"],
        }
    )

    lookup = {
        ("A", "C"): -1.5,
        ("G", "D"): -0.5,
    }

    score = compute_tcren_score(
        pdb_id="test",
        peptide="CD",
        contact_df=contacts,
        lookup=lookup,
    )

    assert score == pytest.approx(-2.0)


def test_compute_tcren_score_uses_requested_pdb():
    """Contacts belonging to other TCR systems should be ignored."""

    contacts = pd.DataFrame(
        {
            "pdb.id": ["test", "other"],
            "residue.index.to": [0, 0],
            "residue.aa.from": ["A", "G"],
        }
    )

    lookup = {
        ("A", "C"): -1.5,
        ("G", "C"): 10.0,
    }

    score = compute_tcren_score(
        pdb_id="test",
        peptide="C",
        contact_df=contacts,
        lookup=lookup,
    )

    assert score == pytest.approx(-1.5)


def test_compute_tcren_score_rejects_missing_contact_map():
    """A missing contact map should raise a clear error."""

    contacts = pd.DataFrame(
        {
            "pdb.id": ["other"],
            "residue.index.to": [0],
            "residue.aa.from": ["A"],
        }
    )

    lookup = {
        ("A", "C"): -1.5,
    }

    with pytest.raises(
        ValueError,
        match="No contact map found",
    ):
        compute_tcren_score(
            pdb_id="test",
            peptide="C",
            contact_df=contacts,
            lookup=lookup,
        )


def test_compute_tcren_score_rejects_invalid_peptide_index():
    """A contact outside the peptide sequence should raise an error."""

    contacts = pd.DataFrame(
        {
            "pdb.id": ["test"],
            "residue.index.to": [3],
            "residue.aa.from": ["A"],
        }
    )

    lookup = {
        ("A", "C"): -1.5,
    }

    with pytest.raises(
        ValueError,
        match="out of range",
    ):
        compute_tcren_score(
            pdb_id="test",
            peptide="C",
            contact_df=contacts,
            lookup=lookup,
        )
