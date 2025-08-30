# ======================================================================================================================
# A script responsible for collecting and storing all data used in this project.
# - Planetary Systems
# - Kaggle: labelled timeseries data
# - Kepler Objects of Interest

# Running this script will overwrite all existing data within the project folder
# ======================================================================================================================

import pandas as pd
import requests
import io

# Construct the query based on what we need, using: https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html
# For the plots
#   1. Exolanet mass vs. orbital period
#   2. Cumulative exoplanet discovery frequency by year
#   3. Eccentricity vs. orbital period
#   4. Planet density vs. planet radius

ps_query = """
SELECT discoverymethod, disc_year, pl_rade, pl_masse, pl_orbper, pl_dens, pl_orbeccen
FROM ps
""" # not worrying about nulls right now

url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
params = {
    "query": ps_query,
    "format": "csv"
}

# send the query to the database
r = requests.get(url, params=params)
df = pd.read_csv(io.StringIO(r.text)) # <- can print(df.head()) if need be to check here

# save the csv
df.to_csv("Plots_Exercise/plots_data.csv", index=False)
print("-> Saved Plots_Exercise/plots_data.csv")

# ======================================================================================================================
# The exoplanet detection data is part of the NASA timeseries data.
# Fortunately, we can find some labelled data easily on Kaggle
# If we end up needing a higher sample size, we can query the timeseries database and cross-reference the lightcurves
# to get our own labelled data.
# But, for now, this will do:

import kagglehub

import os
import shutil
from pathlib import Path

# Define the exoData directory
download_dir = Path(os.path.dirname(__file__)) / "Exoplanet_Detection" / "exoData"
download_dir.mkdir(parents=True, exist_ok=True)

# check if the folder exists, and if so, remove to overwrite:
if download_dir.exists():
    shutil.rmtree(download_dir)

# set the KAGGLEHUB_CACHE to this folder, otherwise the dataset is downloaded elsewhere
os.environ["KAGGLEHUB_CACHE"] = str(download_dir) # this is session-based, not permanent

path = kagglehub.dataset_download("keplersmachines/kepler-labelled-time-series-data")

print("Path to dataset files:", path)

# now, to make things nicer, we want to extract the csvs from:
# exoData\datasets\keplersmachines\kepler-labelled-time-series-data\versions\3
# and just have them sit inside exoData.

csv_names = ["exoTrain.csv", "exoTest.csv"]

for csv in csv_names:
    shutil.move(str(Path(path) / csv), str(download_dir / csv))
    print(f"[*]{csv} has been moved to {download_dir / csv}")

# remove remaining useless folders
shutil.rmtree(download_dir / "datasets", ignore_errors=True)

print("-> Kaggle timeseries data successfully downloaded and placed in exoData")

# ======================================================================================================================
# Kepler KOI table

koi_query = """
SELECT *
FROM kepler_koi
""" # since we don't know what yet to do with the data, we leave the wrangling to the EDA part

# url as before
params = {
    "query": koi_query,
    "format": "csv"
}

# send the query to the database
r = requests.get(url, params=params)
df = pd.read_csv(io.StringIO(r.text)) # <- can print(df.head()) if need be to check here

# save the csv
df.to_csv("Kepler_Candidates/koi_data.csv", index=False)
print("-> Saved Kepler_Candidates/koi_data.csv")