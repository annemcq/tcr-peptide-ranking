# TCR–Peptide Ranking with Sequence Features and TCRen

Can relatively simple sequence features help rank candidate peptides for a T cell receptor?

I built this project after reading the TCRen work by Karnaukhov et al. (2024). TCRen uses a structure-based statistical potential to score TCR–peptide pairs, and I wanted to explore whether combining that signal with features derived directly from peptide and TCR sequences could improve peptide ranking.

## Dataset

I constructed a dataset from 25 TCR systems with available contact maps.

For each TCR, I used its cognate peptide as the positive example and generated 100 negative peptides of the same length, resulting in **2,525 TCR–peptide pairs**.

For every pair, I calculated three groups of features:

- **Peptide features:** amino-acid composition and simple physicochemical properties
- **TCR features:** features derived from the CDR3α and CDR3β sequences
- **TCRen score:** structure-based statistical potential for the TCR–peptide pair

## Models

I compared seven model/feature combinations:

- TCRen alone
- Logistic Regression — peptide features
- Logistic Regression — peptide + TCR features
- Logistic Regression — full feature set
- Random Forest — peptide features
- Random Forest — peptide + TCR features
- Random Forest — full feature set

Because all candidate peptides belonging to the same TCR are related observations, I did not use a standard random row split.

Instead, I split the data by TCR (`pdb_id`) using grouped train/test splits. This ensures that the model is evaluated on TCRs that were not present during training.

To reduce dependence on one particular split, I repeated the evaluation across **30 grouped train/test splits**.

## Evaluation

Since the main task is ranking candidate peptides for each TCR, I focused on ranking metrics as well as standard classification metrics:

- Mean Reciprocal Rank (MRR)
- Top-1 accuracy
- Top-5 accuracy
- Average Precision
- ROC-AUC

### Results

| Model | MRR | Top-1 | Top-5 | Avg. Precision | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| TCRen baseline | 0.33 | 0.15 | 0.49 | 0.22 | 0.91 |
| LogReg peptide | 0.30 | 0.16 | 0.47 | 0.17 | 0.88 |
| LogReg peptide + TCR | 0.30 | 0.16 | 0.46 | 0.15 | 0.87 |
| RF peptide + TCR | 0.48 | 0.35 | 0.63 | 0.35 | 0.88 |
| RF peptide | 0.51 | 0.38 | 0.67 | 0.35 | 0.89 |
| LogReg full | 0.56 | 0.45 | 0.70 | 0.37 | **0.94** |
| **RF full** | **0.62** | **0.50** | **0.77** | **0.49** | 0.93 |

The full Random Forest produced the strongest mean ranking performance overall, while the full Logistic Regression achieved the highest mean ROC-AUC.

In paired comparisons of MRR across the 30 repeated splits, the full Random Forest significantly outperformed TCRen and the peptide-only and peptide+TCR models after Holm–Bonferroni correction. Its advantage over the full Logistic Regression was not statistically significant.

## A problem I found in the dataset

One of the most useful parts of this project came from looking at the feature importance.

Cysteine and histidine frequency appeared among the strongest sequence features. That made me look more closely at how I had generated the negative peptides.

The negatives were produced by sampling amino acids uniformly, whereas real peptides do not follow a uniform amino-acid distribution. This means that the classifier can partly learn to distinguish **real peptide sequences from artificially generated sequences**, rather than learning TCR–peptide recognition alone.

So although the models perform well on this dataset, I would not interpret the reported performance as evidence that the sequence features alone can predict biological TCR specificity at this level.

This was a useful reminder that a model can achieve good evaluation metrics while exploiting a shortcut introduced during dataset construction.

## Held-out TCR example

As an additional test, I completely removed the A6/Tax system (PDB **1AO7**) from training and used the sequence-only Random Forest (rf_peptide_tcr) to rank its 101 candidate peptides. This model uses peptide and TCR features but does not require a TCRen structural score.

The true cognate peptide ranked:

**1st out of 101 candidates**

I treat this as an illustrative example rather than evidence of generalisation, since it represents only one held-out TCR system.

## What I would try next

The first thing I would change is the negative sampling strategy.

Instead of uniformly generated sequences, I would use biologically plausible negatives sampled from real protein sequences, ideally controlling for peptide length and MHC context. This would make it much harder for the model to exploit simple amino-acid composition differences.

I would also like to:

- evaluate on a larger collection of independent TCR systems
- experiment with richer TCR and peptide representations
- compare sequence-based representations with structural information
- investigate whether ranking performance remains after removing the composition shortcut
- structurally remodel candidate peptides rather than reusing the cognate contact map for TCRen scoring

## Repository structure

```text
tcr-peptide-ranking/
├── data/
├── models/
├── notebooks/
├── plots/
├── results/
├── src/
└── tests/
```

The main analysis code is contained in `src/`, while `notebooks/` contains walkthroughs of the results and inference examples.

## Reproducing the analysis

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the data and feature pipeline:

```bash
python src/build_dataset.py
python src/compute_tcren_scores.py
python src/add_peptide_features.py
python src/add_tcr_features.py
```

Run the repeated evaluation and statistical comparison:

```bash
python src/run_evaluation.py
python src/paired_significance_test.py
```

Run the tests:

```bash
pytest tests/
```

## Reference

The TCRen data and scoring approach are based on:

Karnaukhov VK et al.  
**Structure-based prediction of T cell receptor recognition of unseen epitopes using TCRen.**  
*Nature Computational Science* (2024).  
DOI: 10.1038/s43588-024-00653-0

## License

MIT — see `LICENSE`.
