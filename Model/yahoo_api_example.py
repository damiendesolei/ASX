# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 11:55:37 2026

@author: azz
"""

import yfinance as yf

# Download BHP historical trading data
df = yf.download(
    "DRO.AX",
    period="5y",
    auto_adjust=False
)

print(df.head())