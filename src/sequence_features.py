"""
The following is an extraction from the original pipeline 
(src/add_peptide_features.py, src/add_tcr_features.py), showing 
feature-building functions refactored into reusable functions, 
with the aim of applying them to brand-new peptide/TCR sequences at 
inference time (not just for dataset-building).
"""

AA = list("ACDEFGHIKLMNPQRSTVWY")

AA_MOL_WEIGHT = {
    "A": 89.09, "C": 121.16, "D": 133.10, "E": 147.13, "F": 165.19,
    "G": 75.07, "H": 155.16, "I": 131.17, "K": 146.19, "L": 131.17,
    "M": 149.21, "N": 132.12, "P": 115.13, "Q": 146.15, "R": 174.20,
    "S": 105.09, "T": 119.12, "V": 117.15, "W": 204.23, "Y": 181.19,
}

HYDROPHOBIC = set("AVILMFWY")
SMALL = set("AGSTCPDNV")
TINY = set("AGSC")
POSITIVE = set("KRH")
NEGATIVE = set("DE")
POLAR = set("STNQCY")
AROMATIC = set("FWYH")


def build_peptide_features(peptide: str) -> dict:
    """Same feature set as src/add_peptide_features.py's build_feats()."""
    peptide = peptide.upper().strip()
    length = len(peptide)
    if length == 0:
        raise ValueError("peptide sequence is empty")

    feats = {"length": length}

    groups = {
        "hydrophobic": HYDROPHOBIC, "polar": POLAR, "positive": POSITIVE,
        "negative": NEGATIVE, "aromatic": AROMATIC, "small": SMALL, "tiny": TINY,
    }
    for name, group in groups.items():
        feats[f"{name}_frac"] = sum(a in group for a in peptide) / length

    feats["proline_frac"] = peptide.count("P") / length
    feats["glycine_frac"] = peptide.count("G") / length
    feats["mw_est"] = sum(AA_MOL_WEIGHT.get(a, 0.0) for a in peptide)

    for aa in AA:
        feats[f"frac_{aa}"] = peptide.count(aa) / length

    return feats


def _frac_in_set(seq: str, aa_set: set) -> float:
    return sum(a in aa_set for a in seq) / len(seq) if seq else 0.0


def _aa_frac(seq: str, aa: str) -> float:
    return seq.count(aa) / len(seq) if seq else 0.0


def build_tcr_features(cdr3a: str, cdr3b: str) -> dict:
    """Same feature set as src/add_tcr_features.py, for a single (cdr3a, cdr3b) pair."""
    cdr3a = cdr3a.upper().strip()
    cdr3b = cdr3b.upper().strip()

    feats = {
        "cdr3a_len": len(cdr3a),
        "cdr3b_len": len(cdr3b),
        "cdr3a_hydrophobic_frac": _frac_in_set(cdr3a, HYDROPHOBIC),
        "cdr3a_positive_frac": _frac_in_set(cdr3a, POSITIVE),
        "cdr3a_negative_frac": _frac_in_set(cdr3a, NEGATIVE),
        "cdr3a_aromatic_frac": _frac_in_set(cdr3a, AROMATIC),
        "cdr3a_glycine_frac": _aa_frac(cdr3a, "G"),
        "cdr3a_proline_frac": _aa_frac(cdr3a, "P"),
        "cdr3b_hydrophobic_frac": _frac_in_set(cdr3b, HYDROPHOBIC),
        "cdr3b_positive_frac": _frac_in_set(cdr3b, POSITIVE),
        "cdr3b_negative_frac": _frac_in_set(cdr3b, NEGATIVE),
        "cdr3b_aromatic_frac": _frac_in_set(cdr3b, AROMATIC),
        "cdr3b_glycine_frac": _aa_frac(cdr3b, "G"),
        "cdr3b_proline_frac": _aa_frac(cdr3b, "P"),
    }
    return feats


PEP_COLS = [
    "length", "hydrophobic_frac", "polar_frac", "positive_frac", "negative_frac",
    "aromatic_frac", "small_frac", "tiny_frac", "proline_frac", "glycine_frac", "mw_est",
] + [f"frac_{aa}" for aa in AA]

TCR_COLS = [
    "cdr3a_len", "cdr3b_len",
    "cdr3a_hydrophobic_frac", "cdr3a_positive_frac", "cdr3a_negative_frac",
    "cdr3a_aromatic_frac", "cdr3a_glycine_frac", "cdr3a_proline_frac",
    "cdr3b_hydrophobic_frac", "cdr3b_positive_frac", "cdr3b_negative_frac",
    "cdr3b_aromatic_frac", "cdr3b_glycine_frac", "cdr3b_proline_frac",
]

PEP_TCR_COLS = PEP_COLS + TCR_COLS


def build_feature_row(peptide: str, cdr3a: str, cdr3b: str) -> dict:
    """Full feature row (peptide + TCR features) for one candidate, in PEP_TCR_COLS order."""
    row = {}
    row.update(build_peptide_features(peptide))
    row.update(build_tcr_features(cdr3a, cdr3b))
    return {col: row[col] for col in PEP_TCR_COLS}
