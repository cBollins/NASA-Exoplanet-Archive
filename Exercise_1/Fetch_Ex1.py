# Here, we will fetch the necessary data needed csv files.

import pandas as pd
import requests
import io

# Construct the query based on what we need, using: https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html
# For Exercise 1:
#   1. Exolanet mass vs. orbital period
#   2. Cumulative exoplanet discovery frequency by year
#   3. Eccentricity vs. orbital period
#   4. Planet density vs. planet radius

# So we need:

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
df = pd.read_csv(io.StringIO(r.text))
print(df.head())

# save the csv
df.to_csv("Ex1_data.csv", index=False)