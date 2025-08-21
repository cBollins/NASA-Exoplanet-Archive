# NASA Exoplanet Archive EDA

An ongoing EDA exploring the properties of confirmed exoplanets using publicly available data from [NASA's archive](https://exoplanetarchive.ipac.caltech.edu/):

A full walkthrough of the project is given in `Project_Tour.ipynb`, and how each dataset was fetched in `fetch_data.py` -- including a demonstration of a query.

## Project Goals

Two key aims for this project include exploration of the archives, and classification of raw data. The former will include visualisation of exoplanet features, demonstrating domain knowledge; additionally, we aim to construct a random forest to classify exoplanets from the Kepler Objects of Interest database. The latter will see a full classification pipeline, feeding raw, labelled timeseries lightcurves from flagged exoplanet candidates into a full data preprocessing stage and 1d CNN model.

# Contents

- Feature Plots `/Plots_Exercise`.
- Kepler Candidantes `/Kepler_Candidates`.
- Exoplanet Detection `/Exoplanet_Detection`.

---

## Feature Plots

The first exercise is simply recreating some of the plots seen on the first webpage when you follow the link to the [archive](https://exoplanetarchive.ipac.caltech.edu/):
1. Exolanet mass vs. orbital period
2. Cumulative exoplanet discovery frequency by year
3. Eccentricity vs. orbital period
4. Planet density vs. planet radius

Discussion of each plot and its significance can be found on the individual page inside `/Plots_Exercise`.

---

## Kepler Candidates

This section looks at the Kepler Objects of Interest (KOI) database. After exploring the implications of various `NaN` elements, the data was cleaned and split using Sklearn's `train_test_split(X, y, test_size=0.2, stratify=y)`. The idea is to try and predict the `koi_disposition`, or whether the readings of the exoplanet candidate indicate a "FALSE POSITIVE" or "CONFIRMED" exoplanet. A **RandomForestClassifier** was put to the test on the numerical features included in the KOI dataset, achieving $99.3\%$ accuracy, and a perfect negative recall.

---

## Exoplanet Detection

The goal of this section is to classify **raw** stellar lightcurves of flagged exoplanet candidates. This dataset is sourced from Kaggle, and is not up to date. Hence, this section is an introduction to the techniques and ideas used to filter and classify noisy flux readings. Additionaly, this section aims to address how to handle an **imbalanced dataset**; both training and desting data has a *positive:negative* $\sim 1:100$.

Preprocessing pipeline:

1. Zero-mean, unit-variance scaling across the time-domain.
2. Fourier transform the lightcurves
3. Savitzky Golay smoothing

1D CNN Architecture:

- 1d frequency-domain input (3197 values)
- Convolutional layers:
    - Conv1D (1 -> 16 filters, kernel=5, ReLU, MaxPool)
    - Conv1D (16 -> 32 filters, kernel=5, ReLU, MaxPool)
    - Conv1D (32 -> 64 filters, kernel=3, ReLU, MaxPool)
- MaxPool filters half sequence length (overall reduction by a factor of 8)
- Returns pure logits, handle class imbalance with `torch.BCEWithLogitsLoss`

---

# Project Summary

...
