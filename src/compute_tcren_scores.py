"""Compute TCRen scores for all TCR-peptide pairs in the dataset."""

from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parent.parent
DATASET_CSV = BASE / "processed" / "first_dataset_pairs.csv"
CONTACT_MAP_CSV = BASE / "data" / "contact_maps_PDB.csv"
TCREN_POTENTIAL_CSV = BASE / "data" / "TCRen_potential.csv"
OUTPUT_CSV = BASE / "processed" / "data_tcren_score.csv"


def compute_tcren_score(
    pdb_id: str,
    peptide: str,
    contact_df: pd.DataFrame,
    lookup: dict,
) -> float:
    """Compute the TCRen score for one TCR-peptide pair."""

    contacts = contact_df[
        contact_df["pdb.id"] == pdb_id
    ]

    if contacts.empty:
        raise ValueError(
            f"No contact map found for pdb_id '{pdb_id}'"
        )

    peptide = peptide.strip().upper()
    score = 0.0

    for _, contact in contacts.iterrows():
        pos = int(contact["residue.index.to"])
        tcr_aa = contact["residue.aa.from"]

        if not 0 <= pos < len(peptide):
            raise ValueError(
                f"Peptide index {pos} out of range for pdb_id "
                f"'{pdb_id}' (peptide='{peptide}', "
                f"length={len(peptide)})"
            )

        pep_aa = peptide[pos]

        try:
            score += lookup[(tcr_aa, pep_aa)]
        except KeyError as exc:
            raise ValueError(
                f"Missing TCRen value for pair "
                f"({tcr_aa}, {pep_aa})"
            ) from exc

    return score


def main():
    """Load the input data, compute TCRen scores, and save the result."""

    dataset = pd.read_csv(DATASET_CSV)
    contact_maps = pd.read_csv(CONTACT_MAP_CSV)
    tcren_potential = pd.read_csv(TCREN_POTENTIAL_CSV)

    # Standardize identifiers and amino-acid labels.
    dataset["pdb_id"] = (
        dataset["pdb_id"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    dataset["peptide"] = (
        dataset["peptide"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    for col in [
        "pdb.id",
        "residue.aa.from",
        "residue.aa.to",
    ]:
        contact_maps[col] = (
            contact_maps[col]
            .astype(str)
            .str.strip()
        )

    contact_maps["pdb.id"] = (
        contact_maps["pdb.id"]
        .str.lower()
    )

    contact_maps["residue.aa.from"] = (
        contact_maps["residue.aa.from"]
        .str.upper()
    )

    contact_maps["residue.aa.to"] = (
        contact_maps["residue.aa.to"]
        .str.upper()
    )

    for col in [
        "residue.aa.from",
        "residue.aa.to",
    ]:
        tcren_potential[col] = (
            tcren_potential[col]
            .astype(str)
            .str.strip()
            .str.upper()
        )

    # Map each (TCR residue, peptide residue) pair to its TCRen potential.
    tcren_lookup = dict(
        zip(
            zip(
                tcren_potential["residue.aa.from"],
                tcren_potential["residue.aa.to"],
            ),
            tcren_potential["TCRen"],
        )
    )

    print(
        f"TCRen lookup entries: {len(tcren_lookup)}"
    )

    # Keep only contacts in which the target residue belongs to the peptide.
    contact_maps = contact_maps[
        contact_maps["chain.type.to"]
        .astype(str)
        .str.upper()
        == "PEPTIDE"
    ].copy()

    # Score every candidate peptide.
    scores = []

    for idx, row in dataset.iterrows():
        try:
            score = compute_tcren_score(
                pdb_id=row["pdb_id"],
                peptide=row["peptide"],
                contact_df=contact_maps,
                lookup=tcren_lookup,
            )
            scores.append(score)

        except Exception as err:
            raise RuntimeError(
                f"Scoring failed at row {idx} "
                f"(pdb_id={row['pdb_id']}, "
                f"peptide={row['peptide']})"
            ) from err

    dataset["tcren_score"] = scores

    dataset.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print(
        f"Saved scored dataset to: "
        f"{OUTPUT_CSV.resolve()}"
    )


if __name__ == "__main__":
    main()
