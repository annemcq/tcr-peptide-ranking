# ADD TCREN SCORING TO DATASET

# Import necessary libraries
import pandas as pd
from pathlib import Path

# Baseline settings for code
BASE = Path(__file__).resolve().parent.parent
Dataset_csv = BASE / "processed" / "first_dataset_pairs.csv"
Contact_map_csv = BASE / "data" / "contact_maps_PDB.csv"
TCRen_potential = BASE / "data" / "TCRen_potential.csv"
Output = BASE / "processed" / "data_tcren_score.csv"

# Load necessary files
dataset = pd.read_csv(Dataset_csv)
contact_maps = pd.read_csv(Contact_map_csv)
tcren_potential = pd.read_csv(TCRen_potential)

#print("Loaded dataset:", dataset.shape)
#print("Loaded contact maps:", contact_maps.shape)
#print("Loaded TCRen potential:", tcren_pot.shape)
#print()

###############################################################

# Sanitize source data

dataset["pdb_id"] = dataset["pdb_id"].str.strip().str.lower()
dataset["peptide"] = dataset["peptide"].str.strip().str.upper()

for col in ["pdb.id", "residue.aa.from", "residue.aa.to"]:
    contact_maps[col] = contact_maps[col].str.strip()

contact_maps["pdb.id"] = contact_maps["pdb.id"].str.lower()
contact_maps["residue.aa.from"] = contact_maps["residue.aa.from"].str.upper()
contact_maps["residue.aa.to"] = contact_maps["residue.aa.to"].str.upper()

for col in ["residue.aa.from", "residue.aa.to"]:
    tcren_potential[col] = tcren_potential[col].str.strip().str.upper()


# TCRen lookup: (TCR residue, peptide residue) -> potential score
tcren_lookup = dict(
    zip(
        zip(
            tcren_potential["residue.aa.from"],
            tcren_potential["residue.aa.to"],
        ),
        tcren_potential["TCRen"],
    )
)

print(f"TCRen lookup entries: {len(tcren_lookup)}\n")

###############################################################
# Optional filter to preserve only TCR-to-peptide contacts

contact_maps = contact_maps[
    contact_maps["chain.type.to"].astype(str).str.upper() == "PEPTIDE"
].copy()

#print("Filtered contact maps shape:", contact_maps.shape)
#print()


###############################################################
# TCRen score function

def compute_tcren_score(pdb_id: str, peptide: str, contact_df: pd.DataFrame, lookup: dict) -> float:
    # Compute the TCRen score for each PDB-peptide pair
    contacts = contact_df[contact_df["pdb.id"] == pdb_id]

    if contacts.empty:
        raise ValueError(f"No contact map found for pdb_id '{pdb_id}'")

    peptide = peptide.strip().upper()
    score = 0.0

    for _, contact in contacts.iterrows():
        pos = int(contact["residue.index.to"])
        tcr_aa = contact["residue.aa.from"]

        if not 0 <= pos < len(peptide):
            raise ValueError(
                f"Peptide index {pos} out of range for pdb_id '{pdb_id}' "
                f"(peptide='{peptide}', length={len(peptide)})"
            )

        pep_aa = peptide[pos]

        try:
            score += lookup[(tcr_aa, pep_aa)]
        except KeyError:
            raise ValueError(f"Missing TCRen value for pair ({tcr_aa}, {pep_aa})")

    return score

# Score all candidate peptides

scores = []

for idx, row in dataset.iterrows():
    try:
        scores.append(
            compute_tcren_score(
                pdb_id=row["pdb_id"],
                peptide=row["peptide"],
                contact_df=contact_maps,
                lookup=tcren_lookup,
            )
        )
    except Exception as err:
        raise RuntimeError(
            f"Scoring failed at row {idx} "
            f"(pdb_id={row['pdb_id']}, peptide={row['peptide']})"
        ) from err

dataset["tcren_score"] = scores

#print("Scoring complete.")
#print(dataset.head(), "\n")


###############################################################

#Save
dataset.to_csv(Output, index=False)
print(f"Saved scored dataset to: {Output.resolve()}")