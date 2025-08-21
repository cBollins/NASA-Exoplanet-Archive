# NASA Exoplanet Archive EDA

An ongoing EDA exploring the properties of confirmed exoplanets using publicly available data from NASA's archive:

> https://exoplanetarchive.ipac.caltech.edu/

A full walkthrough of the project is given in `Project_Tour.ipynb`, and how each dataset was fetched in `fetch_data.py` -- including a demonstration of a query.

# Contents

- Webpage Plots `/Plots_Exercise`.
- Kepler Candidantes `/Kepler_Candidates`.
- Exoplanet Detection `/Exoplanet_Detection`.

## Webpage Plots

The first exercise is simply recreating some of the plots seen on the first webpage when you follow the link above:
1. Exolanet mass vs. orbital period
2. Cumulative exoplanet discovery frequency by year
3. Eccentricity vs. orbital period
4. Planet density vs. planet radius

## Kepler Candidates

This section looks at the Kepler Objects of Interest (KOI) database. After exploring the implications of various `NaN` elements, the data was cleaned and split using Sklearn's `train_test_split(X, y, test_size=0.2, stratify=y)`. The idea is to try and predict the `koi_disposition`, or whether the readings of the exoplanet candidate indicate a "FALSE POSITIVE" or "CONFIRMED" exoplanet. A **RandomForestClassifier** was put to the test on the numerical features included in the KOI dataset, achieving $99.3\%$ accuracy, and a perfect negative recall.

## Exoplanet Detection

The goal of this section is to classify **raw** stellar lightcurves of flagged exoplanet candidates. This dataset is sourced from Kaggle, and is not up to date. Hence, this section is an introduction to the techniques and ideas used to filter and classify noisy flux readings. Additionaly, this section aims to address how to handle an **imbalanced dataset** -- both training and desting data has a *positive:negative* $\sim 1:100$. 

# Project Summary

...