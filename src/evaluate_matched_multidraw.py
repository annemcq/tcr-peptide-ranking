"""Robustness of matched-negative TCR ranking to independent peptide draws.

Each negative draw regenerates 100 composition-preserving permutations for
each cognate peptide. The same 30 TCR-grouped train/test splits are then used
within each draw. Statistics in the final summary are computed across draw
means (not across all correlated split-level rows).
"""
from __future__ import annotations

import argparse
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.build_matched_dataset import generate_matched_negatives
from src.run_evaluation import (
    FULL_FEATURES, make_logistic_regression, make_random_forest, ranking_metrics,
)
from src.sequence_features import build_peptide_features, build_tcr_features

DEFAULT_OUT = ROOT / "results"
METRICS = ("mrr", "top1", "top5", "roc_auc", "avg_precision")


def load_reference_inputs(features_path: Path, contacts_path: Path, potential_path: Path):
    source = pd.read_csv(features_path)
    source["pdb_id"] = source["pdb_id"].astype(str).str.strip().str.lower()
    source["peptide"] = source["peptide"].astype(str).str.strip().str.upper()
    positives = source.loc[source["label"].eq(1), ["pdb_id", "peptide", "cdr3a", "cdr3b"]].copy()
    if positives["pdb_id"].duplicated().any() or positives.empty:
        raise ValueError("Expected exactly one positive peptide per TCR")
    positives = positives.sort_values("pdb_id").reset_index(drop=True)

    contacts = pd.read_csv(contacts_path)
    contacts = contacts.loc[contacts["chain.type.to"].astype(str).str.upper().eq("PEPTIDE")].copy()
    contacts["pdb.id"] = contacts["pdb.id"].astype(str).str.strip().str.lower()
    contacts["residue.aa.from"] = contacts["residue.aa.from"].astype(str).str.strip().str.upper()
    contact_lookup = {
        pdb: list(zip(group["residue.index.to"].astype(int), group["residue.aa.from"]))
        for pdb, group in contacts.groupby("pdb.id")
    }

    pot = pd.read_csv(potential_path)
    potential = {
        (r["residue.aa.from"].strip().upper(), r["residue.aa.to"].strip().upper()): float(r["TCRen"])
        for _, r in pot.iterrows()
    }
    for r in positives.itertuples(index=False):
        if r.pdb_id not in contact_lookup:
            raise ValueError(f"Missing contact map for {r.pdb_id}")
    return positives, contact_lookup, potential


def make_draw(positives: pd.DataFrame, contacts: dict, potential: dict, draw: int, base_seed: int) -> pd.DataFrame:
    rows = []
    for r in positives.itertuples(index=False):
        seed = base_seed + draw * 1_000_003 + zlib.crc32(r.pdb_id.encode("utf-8")) % 100_000
        negatives = generate_matched_negatives(r.peptide, n_neg=100, seed=seed)
        if len(negatives) != 100 or len(set(negatives)) != 100:
            raise ValueError(f"Incorrect negative count for {r.pdb_id}")
        ref_counts = Counter(r.peptide)
        tcr_feats = build_tcr_features(r.cdr3a, r.cdr3b)
        for label, pep in [(1, r.peptide)] + [(0, x) for x in negatives]:
            if Counter(pep) != ref_counts:
                raise ValueError(f"Composition mismatch: {r.pdb_id}")
            energy = 0.0
            for pos, aa in contacts[r.pdb_id]:
                if not 0 <= pos < len(pep):
                    raise ValueError(f"Out-of-range contact: {r.pdb_id}")
                energy += potential[(aa, pep[pos])]
            rows.append({
                "pdb_id": r.pdb_id, "peptide": pep, "label": label,
                "tcren_score": energy, **build_peptide_features(pep), **tcr_feats,
            })
    df = pd.DataFrame(rows)
    if len(df) != 101 * len(positives):
        raise ValueError("Unexpected candidate count")
    return df


def evaluate_draw(df: pd.DataFrame, draw: int, n_splits: int, split_seed: int, n_trees: int) -> list[dict]:
    results = []
    groups = df["pdb_id"]
    for split in range(n_splits):
        # Reuse the identical TCR-grouped split schedule for each draw.
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=split_seed + split)
        tr, te = next(splitter.split(df, y=df["label"], groups=groups))
        train, test = df.iloc[tr], df.iloc[te].copy()
        if set(train["pdb_id"]) & set(test["pdb_id"]):
            raise AssertionError("TCR leakage between train/test")

        logistic = make_logistic_regression(split_seed + split)
        forest = make_random_forest(split_seed + split)
        forest.set_params(n_estimators=n_trees)
        logistic.fit(train[FULL_FEATURES], train["label"])
        forest.fit(train[FULL_FEATURES], train["label"])

        scores = {
            "tcren_baseline": -test["tcren_score"].to_numpy(),
            "logreg_full": logistic.predict_proba(test[FULL_FEATURES])[:, 1],
            "rf_full": forest.predict_proba(test[FULL_FEATURES])[:, 1],
        }
        for name, values in scores.items():
            test["model_score"] = values
            row = {"draw": draw, "split": split, "model": name,
                   "n_train_tcr": train["pdb_id"].nunique(),
                   "n_test_tcr": test["pdb_id"].nunique(),
                   "roc_auc": roc_auc_score(test["label"], values),
                   "avg_precision": average_precision_score(test["label"], values)}
            row.update(ranking_metrics(test, "model_score"))
            results.append(row)
    return results


def summarise(splits: pd.DataFrame):
    # Each draw contributes one mean for each model, irrespective of its number
    # of split-level rows. SD in final output measures draw-to-draw variability.
    draw_means = splits.groupby(["draw", "model"], as_index=False)[list(METRICS)].mean()
    model_summary = draw_means.groupby("model")[list(METRICS)].agg(["mean", "std"])
    model_summary.columns = [f"{metric}_{stat}_across_draws" for metric, stat in model_summary.columns]
    model_summary = model_summary.reset_index()
    return draw_means, model_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=10)
    parser.add_argument("--splits", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=10000)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--rf-trees", type=int, default=300)
    parser.add_argument("--features", type=Path, default=ROOT / "processed/data_tcren_features_with_tcr.csv")
    parser.add_argument("--contacts", type=Path, default=ROOT / "data/contact_maps_PDB.csv")
    parser.add_argument("--potential", type=Path, default=ROOT / "data/TCRen_potential.csv")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if min(args.draws, args.splits, args.rf_trees) < 1:
        parser.error("--draws, --splits and --rf-trees must be positive")
    pos, contacts, potential = load_reference_inputs(args.features, args.contacts, args.potential)
    all_rows = []
    for draw in range(args.draws):
        dataset = make_draw(pos, contacts, potential, draw, args.base_seed)
        rows = evaluate_draw(dataset, draw, args.splits, args.split_seed, args.rf_trees)
        all_rows.extend(rows)
        partial = pd.DataFrame(rows)
        print(f"Draw {draw + 1}/{args.draws}: " +
              ", ".join(f"{m} MRR {partial.loc[partial.model.eq(m), 'mrr'].mean():.3f}"
                        for m in ("tcren_baseline", "logreg_full", "rf_full")), flush=True)
    splits = pd.DataFrame(all_rows)
    draw_means, summary = summarise(splits)
    args.out.mkdir(parents=True, exist_ok=True)
    splits.to_csv(args.out / "matched_multidraw_splits.csv", index=False)
    draw_means.to_csv(args.out / "matched_multidraw_draw_means.csv", index=False)
    summary.to_csv(args.out / "matched_multidraw_summary.csv", index=False)
    print("\nAcross-draw summary (mean ± sample SD of each draw's grouped-split mean):")
    print(summary.to_string(index=False))
    print(f"Saved results under {args.out.resolve()}")


if __name__ == "__main__":
    main()
