# Dataset — Breast Cancer Wisconsin (Diagnostic) Data Set (WDBC)

This project uses the **Breast Cancer Wisconsin Diagnostic Dataset (WDBC)**,
originally created by Dr. William H. Wolberg, W. Nick Street, and Olvi L.
Mangasarian at the University of Wisconsin.

## Overview

| Property | Value |
|---|---|
| Samples | 569 |
| Features | 30 numeric, real-valued |
| Classes | 2 — Benign (B), Malignant (M) |
| Class balance | ~62.7% benign, ~37.3% malignant |
| Missing values | None |

Each record is computed from a digitized image of a **fine needle aspirate
(FNA)** of a breast mass. The features describe characteristics of the cell
nuclei present in the image.

## Features

Ten real-valued base measurements are computed for each cell nucleus, and
the **mean**, **standard error (`_se`)**, and **"worst"/largest (`_worst`)**
value of each are recorded, giving 10 × 3 = 30 features:

1. `radius` — mean of distances from center to points on the perimeter
2. `texture` — standard deviation of gray-scale values
3. `perimeter`
4. `area`
5. `smoothness` — local variation in radius lengths
6. `compactness` — perimeter² / area − 1.0
7. `concavity` — severity of concave portions of the contour
8. `concave points` — number of concave portions of the contour
9. `symmetry`
10. `fractal dimension` — "coastline approximation" − 1

Target column: `diagnosis` — `M` = malignant (encoded as `1`), `B` = benign
(encoded as `0`).

## How to Get the Data

You have two options — the code in this repository supports **both**
automatically (see `src/data_preprocessing.py`):

### Option 1 — Download the original CSV (recommended for full fidelity with the report)

Download `data.csv` from any of the following public mirrors and place it in
this folder as `dataset/data.csv`:

- **Kaggle:** https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
- **UCI Machine Learning Repository:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

```bash
# Example using the Kaggle CLI (requires `pip install kaggle` and an API token)
kaggle datasets download -d uciml/breast-cancer-wisconsin-data -p dataset/ --unzip
```

Expected columns: `id`, `diagnosis`, `radius_mean`, `texture_mean`, ...,
`fractal_dimension_worst`, `Unnamed: 32` (an empty trailing column present in
the original export — automatically dropped by the preprocessing pipeline).

### Option 2 — Automatic fallback (no download needed)

If `dataset/data.csv` is not present, `src/data_preprocessing.py` automatically
loads the **identical dataset** bundled with scikit-learn
(`sklearn.datasets.load_breast_cancer`), which contains the same 569 samples
and 30 features. This lets you run the entire pipeline **out of the box**
with zero setup:

```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer(as_frame=True)
```

## Citation

> W. H. Wolberg, W. N. Street, and O. L. Mangasarian, "Breast Cancer
> Wisconsin (Diagnostic) Data Set," UCI Machine Learning Repository, 1995.
