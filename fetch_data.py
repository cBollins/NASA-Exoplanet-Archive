# Here, we will fetch the necessary data needed csv files.

import pandas as pd
import requests
import io

# Construct the query based on what we need, using: https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html
# For the plots
#   1. Exolanet mass vs. orbital period
#   2. Cumulative exoplanet discovery frequency by year
#   3. Eccentricity vs. orbital period
#   4. Planet density vs. planet radius

query = """
SELECT discoverymethod, disc_year, pl_rade, pl_masse, pl_orbper, pl_dens, pl_orbeccen
FROM ps
""" # not worrying about nulls right now

url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
params = {
    "query": query,
    "format": "csv"
}

# send the query to the database
r = requests.get(url, params=params)
df = pd.read_csv(io.StringIO(r.text)) # <- can print(df.head()) if need be to check here

# save the csv
df.to_csv("Plots_Exercise/plots_data.csv", index=False)

# ======================================================================================================================
# The exoplanet detection data is part of the NASA timeseries data.
# Fortunately, we can find some labelled data easily on Kaggle
# If we end up needing a higher sample size, we can query the timeseries database and and cross-reference the lightcurves
# to get our own labelled data.
# But, for now, this will do:

import kagglehub
import os

filepath = "Exoplanet_Detection/exoData"

path = kagglehub.dataset_download("keplersmachines/kepler-labelled-time-series-data", path=filepath)
print("Path to dataset files:", path)