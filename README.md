# NASA Exoplanet Archive EDA

An ongoing EDA exploring the properties of confirmed exoplanets using publicly available data from [NASA's archive](https://exoplanetarchive.ipac.caltech.edu/):

A full walkthrough of the project is given in `Project_Tour.ipynb`, and data gathering in `fetch_data.py` &mdash; you can run this script to access most of the data too.

## Project Goals

Two key aims for this project include exploration of the archives, and classification of raw data. The former will include visualisation of exoplanet features, demonstrating domain knowledge; additionally, we aim to construct a random forest to classify exoplanets from the Kepler Objects of Interest database. The latter will see a full classification pipeline, feeding raw, labelled lightcurves from flagged exoplanet candidates into a full data preprocessing stage and 1D CNN model.

# Contents

- Exoplanet feature plots `Plots_Exercise/`.
- Kepler KOI candidates table `Kepler_Candidates/`.
- Lightcurves `Exoplanet_Detection/`.
- Images & plots `Media/`.

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

This section explores the Kepler Objects of Interest (KOI) database. After exploring the implications of various `NaN` elements, the data was cleaned and split using Sklearn's `train_test_split(X, y, test_size=0.2, stratify=y)`. The idea is to try and predict the `koi_disposition`, or whether the readings of the exoplanet candidate indicate a "FALSE POSITIVE" or "CONFIRMED" exoplanet.

An Sklearn **RandomForestClassifier** was put to the test on the numerical features included in the KOI dataset, achieving 99.3\% accuracy, and a perfect negative recall.

---

## Exoplanet Detection

The goal of this section is to classify **raw** stellar lightcurves of flagged exoplanet candidates. This dataset is sourced from Kaggle, and is not up to date. Hence, this section is an introduction to the techniques and ideas used to filter and classify noisy flux readings. Additionaly, this section aims to address how to handle an **imbalanced dataset**; both training and desting data has a ratio positive:negative $\sim 1:100$.

### Preprocessing pipeline:

1. Zero-mean, unit-variance scaling across the time-domain &mdash; commonly called Z-score normalisation.
2. Fourier transform the lightcurves from the time-domain to the frequency-domain.
3. Savitzky Golay filter, to smooth the frequency data.

### 1D CNN Architecture:

- 1d frequency-domain input, shape: (3197,).
- Convolutional layers:
    - Conv1D (1 &rightarrow; 16 filters, kernel=5, ReLU, MaxPool)
    - Conv1D (16 &rightarrow; 32 filters, kernel=5, ReLU, MaxPool)
    - Conv1D (32 &rightarrow; 64 filters, kernel=3, ReLU, MaxPool)
- MaxPool filters half sequence length (overall reduction by a factor of 8)
- Returns pure logits, handle class imbalance with `torch.BCEWithLogitsLoss`

### Training and Performance on Kaggle dataset:

The ExoCNN object was initialised and run over 12 epochs using the `nn.BCEWithLogitsLoss()` loss function, which is more stable than boostrapping a sigmoid inside the preprocessing instance with a BCELoss function. This also gives more headroom to make improvements to the class-imbalance issue we are facing later. The model finished training with a running `Epoch [12/12] - Loss: 0.0039`.

The model, as expected, returned an extremely high accuracy of 99.5\%. What we really want to focus on is the recall, though. If we treat the positive class (flagging an exoplanet as True) as binomial:

$$
Recall = \frac{TP}{TP + FN} \sim B(5,0.6)
$$

- Each exoplanet has a 60% chance of being classed as true.
- Classifications are independent.
- We have 5 exoplanets in the training data.

And we make use of the Clopper-Pearson confidence interval with a two-tailed binomial distribution:

$$
p_{min} = Beta^{-1}(\frac\alpha2, x, n-x+1) $$ $$
p_{max} = Beta^{-1}(1-\frac\alpha2, x+1, n-x)
$$

Where:
- $x$ is the number of successes, $x=TP$.
- $n$ is the number of trials, $n=TP+FN$.
- $\alpha$ is our significance level.

Here, $\alpha=5$ gives a 95\% ($1-\alpha$) confidence interval $[p_{min}, p_{max}]$ = $[0.147, 0.947]$.

In short, even though we observed a 60\% recall, our true recall could plausibly be anywhere between 0.15 and 0.95.

### Why This DOESN'T MATTER

Yes, having a largely imbalanced dataset with only 5 positives does suck and isn't very instructive, the point of this exercise was to build a preprocessing pipeline, bootstrap it to a CNN architecture, and test the process. While we *literally &mdash; statistically* &mdash; don't know whether to be impressed by the result, we now have the tools to manage and classify future labelled lightcurves. While numerical metrics are unreliable on such a tiny positive sample, the workflow is now validated and ready to scale to both larger and better balanced datasets.

### Considering the Next Steps

The preprocessing went fairly smoothly, but one thing to note is that perhaps not all lightcurves will have the same shape. This releases the potential of exploring resampling or interpolation algorithms, perhaps padding or truncating some. It must be mentioned that resampling/padding could affect frequency content during the FFT.

The challenges before was size, but now as we source and cross-reference various light curves from different surveys, we may need to consider the differing equipment used to measure flux. The pipeline may not be suited to discerning between potentially significantly different types of data.

Handling class imbalance is somewhat of a buzzword we haven't actually addressed yet, primarily due to the low ROI due to the statistically limited positive samples. Here are a few ways we can experiment on a new dataset to incentivise positives in the ExoCNN:
1. **Adjust the decision threshold at model inference.** In the snippet `preds = (outputs >= 0.5).int()`, we are asking the model to check whether it is *more* sure of a positive or a negative, and then we just take whichever one it prefers. Potentially adjusting from 0.5 &rightarrow; 0.25 (or similar) we could try and catch more positives.
2. **Adding a positive weight to the BCE Loss function,** making the model more sensitive to positives.
3. **Focal Loss** downweights easy negatives and focusses on rare, more 'difficult' positives.

    $$
    FL(p_t) = -\alpha(1-p_t)^\gamma\log p_t
    $$

    We can write a loss function using the `torch.nn.functional` package by composing a BCEWithLogits() function with $FL(p_t)$.
   
4. **Oversample positives in the DataLoader** using PyTorch's `torch.utils.data.WeightedRandomSampler`. This doesn't discard negatives and reduce sample size, but merely increase the change that a positive is sampled multiple times per epoch. The model doesn't see it the same though, oversampling increases the gradient updates during training.

These are heuristics for low-sample regimes; they don't actually change the sample size, more change how the model 'feels' about finding them. These are practical tools, not magic fixes, and a deep set of clean, distinct, statistically consistent data will *always* be essential when training these kinds of models.

---

# Project Summary

Project is ongoing, there is nothing significant to summarise so far that cannot be read above. Please be patient and await further conclusions!