"""Prepare the composition-matched dataset for model evaluation."""

from pathlib import Path
import sys

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.compute_tcren_scores import compute_tcren_score
from src.sequence_features import build_peptide_features, build_tcr_features


INPUT_CSV = BASE / "processed" / "matched_composition_pairs.csv"
CONTACT_MAP_CSV = BASE / "data" / "contact_maps_PDB.csv"
TCREN_POTENTIAL_CSV = BASE / "data" / "TCRen_potential.csv"
TCR_SOURCE_CSV = BASE / "data" / "summary_PDB_structures.csv"
OUTPUT_CSV = BASE / "processed" / "data_tcren_features_matched_composition.csv"


def main():
    """Add TCRen, peptide features, and TCR features to matched pairs."""
    df = pd.read_csv(INPUT_CSV)
    contacts = pd.read_csv(CONTACT_MAP_CSV)
    potential = pd.read_csv(TCREN_POTENTIAL_CSV)
    tcr_source = pd.read_csv(TCR_SOURCE_CSV)

    df["pdb_id"] = df["pdb_id"].astype(str).str.strip().str.lower()
    df["peptide"] = df["peptide"].astype(str).str.strip().str.upper()

    contacts["pdb.id"] = contacts["pdb.id"].astype(str).str.strip().str.lower()
    contacts["residue.aa.from"] = contacts["residue.aa.from"].astype(str).str.strip().str.upper()
    contacts["residue.aa.to"] = contacts["residue.aa.to"].astype(str).str.strip().str.upper()
    contacts = contacts[
        contacts["chain.type.to"].astype(str).str.upper() == "PEPTIDE"
    ].copy()

    potential["residue.aa.from"] = potential["residue.aa.from"].astype(str).str.strip().str.upper()
    potential["residue.aa.to"] = potential["residue.aa.to"].astype(str).str.strip().str.upper()

    lookup = dict(
        zip(
            zip(potential["residue.aa.from"], potential["residue.aa.to"]),
            potential["TCRen"],
        )
    )

    scores = []
    for _, row in df.iterrows():
        scores.append(
            compute_tcren_score(
                row["pdb_id"],
                row["peptide"],
                contacts,
                lookup,
            )
        )
    df["tcren_score"] = scores

    tcr_source["pdb.id"] = tcr_source["pdb.id"].astype(str).str.strip().str.lower()
    tcr_sequences = (
        tcr_source[["pdb.id", "cdr3a", "cdr3b"]]
        .rename(columns={"pdb.id": "pdb_id"})
        .drop_duplicates("pdb_id")
    )

    df = df.merge(
        tcr_sequences,
        on="pdb_id",
        how="left",
        validate="many_to_one",
    )

    if df[["cdr3a", "cdr3b"]].isna().any().any():
        raise ValueError("Missing CDR3 sequences in matched dataset.")

    peptide_features = pd.DataFrame(
        [build_peptide_features(p) for p in df["peptide"]],
        index=df.index,
    )
    tcr_features = pd.DataFrame(
        [build_tcr_features(a, b) for a, b in zip(df["cdr3a"], df["cdr3b"])],
        index=df.index,
    )

    df = pd.concat([df, peptide_features, tcr_features], axis=1)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Prepared {len(df)} matched TCR-peptide pairs.")
    print(f"Saved: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
