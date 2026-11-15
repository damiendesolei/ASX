# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 23:18:32 2026

@author: azz
"""


import yfinance as yf
import requests

import pandas as pd

import io
import time


DATA_PATH = r"F:\AXS\20261115\\"

# # Download DRO historical trading data
# df = yf.download(
#     "DRO.AX",
#     period="6mo",
#     auto_adjust=False
# )
# df.columns = df.columns.get_level_values(0).str.lower()
# # print(df.head())

ASX_LIST_URL = "https://www.asx.com.au/asx/research/ASXListedCompanies.csv"


def get_asx_listing() -> pd.DataFrame:
    """Listed ASX companies with a Yahoo-style ticker column (e.g. DRO.AX)."""
    resp = requests.get(ASX_LIST_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    lines = resp.text.splitlines()
    header_row = next(i for i, line in enumerate(lines) if "asx code" in line.lower())
    listing = pd.read_csv(io.StringIO("\n".join(lines[header_row:])))
    listing.columns = listing.columns.str.strip().str.lower().str.replace(" ", "_")
    listing["asx_code"] = listing["asx_code"].astype(str).str.strip()
    listing["ticker"] = listing["asx_code"] + ".AX"
    return listing.drop_duplicates("ticker").reset_index(drop=True)



def download_prices(tickers, period="1y", batch_size=25, pause=10.0, max_retries=3):
    frames = []
    for start in range(0, len(tickers), batch_size):
        batch = tickers[start:start + batch_size]
        for attempt in range(1, max_retries + 1):
            raw = yf.download(batch, period=period, auto_adjust=False,
                              group_by="ticker", threads=False, progress=False)
            if raw is not None and not raw.dropna(how="all").empty:
                break
            time.sleep(300 * attempt)  # rate-limited: wait 5, 10, 15 min
        else:
            continue
        long = raw.stack(level=0, future_stack=True).rename_axis(["date", "ticker"])
        long.columns = long.columns.str.lower().str.replace(" ", "_")
        frames.append(long.dropna(how="all").reset_index())
        time.sleep(pause)
    prices = pd.concat(frames, ignore_index=True).dropna(subset=["close"])
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    return prices, sorted(set(tickers) - set(prices["ticker"]))


# Example usage:
listing = get_asx_listing()
prices, missing = download_prices(listing["ticker"].tolist(), period="1yr")
print(prices.shape, f"{len(missing)} tickers returned no data")


# Retry the ones that returned nothing (some will be suspended or not on Yahoo)
# if missing:
#     retry, missing = download_prices(missing, period="6mo", batch_size=25, pause=5.0)
#     prices = pd.concat([prices, retry], ignore_index=True).sort_values(["ticker", "date"])

prices.to_parquet(DATA_PATH + "asx_prices.parquet", index=False)