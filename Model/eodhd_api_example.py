# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 12:06:08 2026

@author: azz
"""

import requests
import pandas as pd

# EODHD API credentials
api_key = "6af26d46d69eb0.38579596"

# ASX ticker (BHP Group)
ticker = "DRO.AU"

# API endpoint
url = f"https://eodhd.com/api/eod/{ticker}"

# Parameters
params = {
    "api_token": api_key,
    "from": "2020-01-01",
    "to": "2026-11-10",
    "period": "d",
    "fmt": "json"
}

# Request data
response = requests.get(url, params=params, timeout=30)
response.raise_for_status()

# Convert to Pandas DataFrame
df = pd.DataFrame(response.json())

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# Set date as index
df = df.set_index("date").sort_index()

# Display stock prices and trading volume
print(df.head())