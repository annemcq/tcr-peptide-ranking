# PLOT RESULTS

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
results_csv = BASE / "results" / "repeated_eval_all_results_with_tcr.csv"
summary_csv = BASE / "results" / "repeated_eval_summary_with_tcr.csv"

df = pd.read_csv(results_csv)
summary_df = pd.read_csv(summary_csv)

model_order = [
    "tcren_baseline",
    "logreg_peptide",
    "rf_peptide",
    "logreg_peptide_tcr",
    "rf_peptide_tcr",
    "logreg_full",
    "rf_full",
]

label_map = {
    "tcren_baseline": "TCRen",
    "logreg_peptide": "LogReg peptide",
    "rf_peptide": "RF peptide",
    "logreg_peptide_tcr": "LogReg peptide+TCR",
    "rf_peptide_tcr": "RF peptide+TCR",
    "logreg_full": "LogReg full",
    "rf_full": "RF full",
}

df = df[df["model"].isin(model_order)].copy()
df["model_label"] = df["model"].map(label_map)

summary_df = summary_df[summary_df["model"].isin(model_order)].copy()
summary_df["model_label"] = summary_df["model"].map(label_map)

ordered_labels = [label_map[m] for m in model_order]

# MRR boxplot
plt.figure(figsize=(10, 5))
data = [df.loc[df["model"] == m, "mrr"].values for m in model_order]
plt.boxplot(data, tick_labels=ordered_labels)
plt.xticks(rotation=25, ha="right")
plt.ylabel("MRR")
plt.title("MRR across repeated grouped splits")
plt.tight_layout()
plt.savefig(BASE / "plots" / "mrr_boxplot.png", dpi=300)
plt.close()

# Top-1 boxplot
plt.figure(figsize=(10, 5))
data = [df.loc[df["model"] == m, "top1"].values for m in model_order]
plt.boxplot(data, tick_labels=ordered_labels)
plt.xticks(rotation=25, ha="right")
plt.ylabel("Top-1")
plt.title("Top-1 success across repeated grouped splits")
plt.tight_layout()
plt.savefig(BASE / "plots" / "top1_boxplot.png", dpi=300)
plt.close()

# Average precision boxplot
plt.figure(figsize=(10, 5))
data = [df.loc[df["model"] == m, "avg_precision"].values for m in model_order]
plt.boxplot(data, tick_labels=ordered_labels)
plt.xticks(rotation=25, ha="right")
plt.ylabel("Average precision")
plt.title("Average precision across repeated grouped splits")
plt.tight_layout()
plt.savefig(BASE / "plots" / "avg_precision_boxplot.png", dpi=300)
plt.close()

# Top-1 mean barplot with error bars
summary_df = summary_df.set_index("model").loc[model_order].reset_index()

plt.figure(figsize=(10, 5))
plt.bar(
    summary_df["model_label"],
    summary_df["top1_mean"],
    yerr=summary_df["top1_std"],
    capsize=4
)
plt.xticks(rotation=25, ha="right")
plt.ylabel("Top-1 mean")
plt.title("Mean Top-1 performance across repeated grouped splits")
plt.tight_layout()
plt.savefig(BASE / "plots" / "summary_top1_barplot.png", dpi=300)
plt.close()

print("Saved plots to:")
print(BASE / "plots" / "mrr_boxplot.png")
print(BASE / "plots" / "top1_boxplot.png")
print(BASE / "plots" / "avg_precision_boxplot.png")
print(BASE / "plots" / "summary_top1_barplot.png")