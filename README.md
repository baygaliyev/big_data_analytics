# Big Data Analytics — Car Crash Prediction

Big Data Analytics coursework project (2019/2020), MSc *Data Science and
Business Informatics*, University of Pisa.

**Goal:** predict whether a private vehicle will be involved in a car crash in
the month *after* the months its mobility data refers to — and predict it with
**high recall**, because missing a future crash is far more expensive than a
false alarm.

## Dataset

Individual mobility networks for ~**5,000 private vehicles** over **four months**,
built from on-board GPS devices pinging every 30 s over GPRS. Roughly **400
features** per vehicle describe three areas of the mobility network — the same
schema for **London, Rome and Tuscany**.

- 21 train + 21 test sets (7 per city)
- Tuscany: 429 attributes; London: 2 fewer; Rome: 6 fewer
- Crash records identify the vehicles involved, giving the binary target

The dataset is **not included in this repository** (size and licensing). Drop
the CSVs into `data/` — the notebooks resolve them from there (override with
the `COURSE_DATA_DIR` environment variable).

## What's here

| File | Contents |
|---|---|
| `bda_1_clustering.ipynb` | Standalone clustering exercise: K-Means, DBSCAN, Hierarchical, elbow + silhouette validation |
| `bda_2_classification.ipynb` | Crash prediction: unified 3-city training set and per-city testing |
| `bda_3_final_part.ipynb` | Final runs: clustered variants + test-set evaluation on all three cities |
| `bda_report_Aliyev_Barandoni_Fanelli_Parshina.pdf` | Submitted report |

## Method

- **Class balancing** — for each place, records are drawn from the seven
  same-place datasets; the working frame pairs every crash row with a random
  sample of non-crash rows.
- **Clustering as a feature** — hierarchical clustering (k = 8) is added as a
  cluster-label column. It did not appear among the important features of the
  decision tree and was dropped again.
- **Classifiers tried** — **Decision Tree** and **Random Forest** as the main
  approaches, each best tree boosted with **AdaBoost**; **k-NN** performed
  badly; **PCA** did not improve any classifier; clustering labels did not help.
- **Generalisation** — models are trained per city and on the pooled 3-city
  dataset, then evaluated on held-out city test sets.

## Result

The pooled 3-city model was the only one that transferred well to unseen
territory. Best tree ≈ boosted decision tree; Random Forest landed at roughly
the same place. The trade-off is complexity: sub-trees reach 40–45 levels of
depth, which makes the model hard to interrogate at the level of individual
crashed records — and it needs a large, specific feature set to rebuild an
individual mobility network per vehicle.

## Reproducing

```bash
pip install pandas numpy matplotlib scikit-learn
jupyter notebook bda_1_clustering.ipynb   # needs credit_default_train.csv
jupyter notebook bda_2_classification.ipynb
jupyter notebook bda_3_final_part.ipynb
```

## Team

Gurban Aliyev · Simone Barandoni · Gioele Fanelli · Kseniia Parshina —
University of Pisa, A.A. 2019/2020.
