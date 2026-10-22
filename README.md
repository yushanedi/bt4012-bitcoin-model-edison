# Fraudulent Bitcoin Transaction Detection

A leak-free, regime-aware ML pipeline for classifying Bitcoin transactions as illicit or licit on an Elliptic-style transaction graph. Full write-up: [`report.pdf`](report.pdf).

## Task

Classify each of 157,101 transactions (234,355 directed payment-flow edges, 49 `time_step`s) as illicit (1) or licit (0) from 165 anonymized features (94 local + 71 neighborhood aggregates). Only 31,235 transactions are labeled (11.7% illicit). `train.csv` covers `time_step`s 1–35; `test.csv` covers `time_step`s 36–49 and carries **no labels available to us** — predictions for it are the deliverable (`submission.csv`), scored externally by ROC-AUC.

## Files

| File | Description |
|---|---|
| `analysis.ipynb` | Full pipeline: EDA, baselines (Logistic Regression, GCN, GAT, GIN, LightGBM), leak-free graph-structural feature engineering, domain-drift correction, horizon-matched rolling-origin CV, feature-count reduction, a seed-bagged XGBoost reference model, and the final ensemble (neighbour-score stacking, unlabeled train rows as weak licit examples, rank-averaged XGBoost / ExtraTrees / Random Forest members over several feature views). Run top to bottom from the project root (`DATA_DIR = '.'`). |
| `train.csv` | Training data, `time_step`s 1–35 (141,772 rows; 31,235 labeled). |
| `test.csv` | Scoring data, `time_step`s 36–49 (15,329 rows, unlabeled). |
| `txs_edgelist.csv` | Directed transaction graph edges (234,355 rows), joins `train.csv`/`test.csv` via `txId`. |
| `sample_submission.csv` | Required submission format (`index`, `target`). |
| `submission.csv` | Final predictions — `target` is the final ensemble's rank-averaged illicit score for each `test.csv` row (higher = more likely illicit). |
| `report.pdf` | 4-page technical report: methodology, results, and reflection. |

## Key result

Because `test.csv` has no labels available to us, no test ROC-AUC is computed anywhere in this repo — every number is a validation score derived entirely from `train.csv`. The central finding is a **validation-methodology effect**: a naive near-split validation (last ~15% of labeled `time_step`s) reports an overoptimistic **0.998** ROC-AUC, while horizon-matched rolling-origin CV — designed to match the real, much larger training→scoring gap — settles at a more defensible **≈0.89**, recovered to **≈0.92** after aggressive feature-count reduction (172→44 features) and 10-seed bagging. See `report.pdf` §3–4 for the full analysis.

## Reproducing

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn networkx lightgbm xgboost torch torch-geometric
jupyter nbconvert --to notebook --execute analysis.ipynb
```

Running the notebook end-to-end regenerates `submission.csv` from the current `train.csv`/`test.csv`/`txs_edgelist.csv`.

## Note on data leakage

No model is trained on `test.csv` labels — every training label comes from `train.csv` (labeled rows, plus unlabeled rows entered as weak licit examples). `test.csv` is read only for its features: to compute per-`time_step` graph and rank features, to form neighbour-score features, and to receive predictions.
