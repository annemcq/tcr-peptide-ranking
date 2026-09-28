# MODEL 1 EVALUATION

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score


BASE = Path(__file__).resolve().parent.parent
Source_CSV = BASE / "processed" / "data_tcren_features_with_tcr.csv"
Results_CSV = BASE / "results" / "repeated_eval_all_results_with_tcr.csv"
Summary_CSV = BASE / "results" / "repeated_eval_summary_with_tcr.csv"

# Required settings
n_repeats = 30
Size = 0.30
R_S = 42

###############################################################
# Load data

df = pd.read_csv(Source_CSV)

needed = ["pdb_id", "peptide", "label", "tcren_score", "cdr3a", "cdr3b"]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df["pdb_id"] = df["pdb_id"].astype(str).str.strip().str.lower()
df["peptide"] = df["peptide"].astype(str).str.strip().str.upper()

###############################################################
# Features

aa_cols = [c for c in df.columns if c.startswith("frac_")]

pep_cols = [
    "length",
    "hydrophobic_frac",
    "polar_frac",
    "positive_frac",
    "negative_frac",
    "aromatic_frac",
    "small_frac",
    "tiny_frac",
    "proline_frac",
    "glycine_frac",
    "mw_est",
] + aa_cols

tcr_cols = [
    "cdr3a_len",
    "cdr3b_len",
    "cdr3a_hydrophobic_frac",
    "cdr3a_positive_frac",
    "cdr3a_negative_frac",
    "cdr3a_aromatic_frac",
    "cdr3a_glycine_frac",
    "cdr3a_proline_frac",
    "cdr3b_hydrophobic_frac",
    "cdr3b_positive_frac",
    "cdr3b_negative_frac",
    "cdr3b_aromatic_frac",
    "cdr3b_glycine_frac",
    "cdr3b_proline_frac",
]

missing = [c for c in pep_cols + tcr_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing feature columns: {missing}")

pep_tcr_cols = pep_cols + tcr_cols
full_cols = pep_cols + tcr_cols + ["tcren_score"]

results = []

###############################################################
# Repeated grouped train/test split + model evaluation

for i in range(n_repeats):
    split_seed = R_S + i

    splitter = GroupShuffleSplit(n_splits=1, test_size=Size, random_state=split_seed)
    train_idx, test_idx = next(splitter.split(df, y=df["label"], groups=df["pdb_id"]))

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    overlap = set(train_df["pdb_id"]).intersection(set(test_df["pdb_id"]))
    if overlap:
        raise ValueError(f"Overlapping pdb_ids found: {sorted(overlap)}")

    X_train_pep = train_df[pep_cols]
    X_test_pep = test_df[pep_cols]

    X_train_pep_tcr = train_df[pep_tcr_cols]
    X_test_pep_tcr = test_df[pep_tcr_cols]

    X_train_full = train_df[full_cols]
    X_test_full = test_df[full_cols]

    y_train = train_df["label"].values
    y_test = test_df["label"].values

    ###############################################################
    # Models

    logreg_pep = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=split_seed))
    ])

    rf_pep = RandomForestClassifier(
        n_estimators=300,
        random_state=split_seed,
        class_weight="balanced"
    )

    logreg_pep_tcr = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=split_seed))
    ])

    rf_pep_tcr = RandomForestClassifier(
        n_estimators=300,
        random_state=split_seed,
        class_weight="balanced"
    )

    logreg_full = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=split_seed))
    ])

    rf_full = RandomForestClassifier(
        n_estimators=300,
        random_state=split_seed,
        class_weight="balanced"
    )

    logreg_pep.fit(X_train_pep, y_train)
    rf_pep.fit(X_train_pep, y_train)

    logreg_pep_tcr.fit(X_train_pep_tcr, y_train)
    rf_pep_tcr.fit(X_train_pep_tcr, y_train)

    logreg_full.fit(X_train_full, y_train)
    rf_full.fit(X_train_full, y_train)

    ###############################################################
    # Scores and metrics

    test_df = test_df.copy()
    test_df["score_tcren"] = -test_df["tcren_score"]

    test_df["score_logreg_pep"] = logreg_pep.predict_proba(X_test_pep)[:, 1]
    test_df["score_rf_pep"] = rf_pep.predict_proba(X_test_pep)[:, 1]

    test_df["score_logreg_pep_tcr"] = logreg_pep_tcr.predict_proba(X_test_pep_tcr)[:, 1]
    test_df["score_rf_pep_tcr"] = rf_pep_tcr.predict_proba(X_test_pep_tcr)[:, 1]

    test_df["score_logreg_full"] = logreg_full.predict_proba(X_test_full)[:, 1]
    test_df["score_rf_full"] = rf_full.predict_proba(X_test_full)[:, 1]

    score_cols = {
        "tcren_baseline": "score_tcren",
        "logreg_peptide": "score_logreg_pep",
        "rf_peptide": "score_rf_pep",
        "logreg_peptide_tcr": "score_logreg_pep_tcr",
        "rf_peptide_tcr": "score_rf_pep_tcr",
        "logreg_full": "score_logreg_full",
        "rf_full": "score_rf_full",
    }

    for model_name, col in score_cols.items():
        scores = test_df[col].values
        preds = (scores >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_test, scores)
        avg_precision = average_precision_score(y_test, scores)
        accuracy = accuracy_score(y_test, preds)

        ranks = []

        for pdb_id, group in test_df.groupby("pdb_id"):
            group = group.sort_values(col, ascending=False).reset_index(drop=True)
            pos_idx = np.where(group["label"].values == 1)[0]

            if len(pos_idx) != 1:
                raise ValueError(f"{pdb_id}: expected 1 positive, got {len(pos_idx)}")

            ranks.append(int(pos_idx[0]) + 1)

        ranks = np.array(ranks)

        results.append({
            "repeat": i,
            "model": model_name,
            "n_train_groups": train_df["pdb_id"].nunique(),
            "n_test_groups": test_df["pdb_id"].nunique(),
            "roc_auc": roc_auc,
            "avg_precision": avg_precision,
            "accuracy": accuracy,
            "mean_rank": ranks.mean(),
            "median_rank": np.median(ranks),
            "mrr": np.mean(1.0 / ranks),
            "top1": np.mean(ranks == 1),
            "top5": np.mean(ranks <= 5),
        })

###############################################################
# Save results

results_df = pd.DataFrame(results)
results_df.to_csv(Results_CSV, index=False)

metrics = ["roc_auc", "avg_precision", "accuracy", "mean_rank", "median_rank", "mrr", "top1", "top5"]

summary_rows = []

for model_name, g in results_df.groupby("model"):
    row = {"model": model_name}
    for metric in metrics:
        row[f"{metric}_mean"] = g[metric].mean()
        row[f"{metric}_std"] = g[metric].std()
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(Summary_CSV, index=False)

#print(summary_df.sort_values("mrr_mean", ascending=False))