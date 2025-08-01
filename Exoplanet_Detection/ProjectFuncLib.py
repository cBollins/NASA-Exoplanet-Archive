import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft, fftfreq, rfft
from sklearn.model_selection import train_test_split

# create a dictionary based on label:
#   2 = at least one exoplanet, and hence will appear blue.
#   1 = no exoplanets in the stars orbit, and the colour will be red
c_label = {
    2: "blue",
    1: "red"
}

label_names = {
    2: "Exoplanet",
    1: "Non-Exoplanet"    
}

def scale_rows(X):
    # since sklearn's StandardScaler works by column, we do this manually.
    means = X.mean(axis=1)
    stdvs = X.std(axis=1)

    # return scaled rows,
    # X_i |-> (X_i - mu) / sigma
    return X.sub(means, axis=1).div(stdvs, axis=1) # vectorised operations

def fft_lc(X):
    return np.abs(rfft(X, axis=1))

def preprocess(df, target_col="LABEL", **tts_args): # returns X_train, X_test, y_train, y_test as a tuple
    X = fft_lc(scale_rows(df.drop([target_col], axis=1)))
    y = df[target_col]
    return train_test_split(X, y, **tts_args)