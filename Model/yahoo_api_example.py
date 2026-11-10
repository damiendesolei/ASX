# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 11:55:37 2026

@author: azz
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Download DRO historical trading data
df = yf.download(
    "DRO.AX",
    period="1y",
    auto_adjust=False
)
df.columns = df.columns.get_level_values(0).str.lower()
# print(df.head())


def add_macd(df, fast=12, slow=26, signal=9, price_col='close'):
    """
    计算 MACD，返回在原 df 基础上新增 DIF / DEA / MACD 三列的新 DataFrame
    参数默认 (12, 26, 9)，与通达信/同花顺一致
    """
    df = df.copy()

    # 1. 快慢 EMA —— 关键：adjust=False 才是递推式 EMA，与国内软件一致
    ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()

    # 2. DIF（快线）
    df['DIF'] = ema_fast - ema_slow

    # 3. DEA（慢线 / 信号线）
    df['DEA'] = df['DIF'].ewm(span=signal, adjust=False).mean()

    # 4. MACD 柱 —— 国内软件习惯 ×2
    df['MACD'] = (df['DIF'] - df['DEA']) #* 2

    return df

# 假设 df 的索引是 DatetimeIndex，列至少包含 close
df = add_macd(df)


def add_macd_signals(df):
    df = add_macd(df)
    # DIF 上穿 DEA = 金叉（1），下穿 = 死叉（-1）
    cross_up   = (df['DIF'] > df['DEA']) & (df['DIF'].shift(1) <= df['DEA'].shift(1))
    cross_down = (df['DIF'] < df['DEA']) & (df['DIF'].shift(1) >= df['DEA'].shift(1))

    df['signal'] = 0
    df.loc[cross_up,   'signal'] = 1
    df.loc[cross_down, 'signal'] = -1
    return df

df = add_macd_signals(df)
#print(df[['close', 'DIF', 'DEA', 'MACD']].tail())


### Print ###
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})
ax1.plot(df.index, df['close'], label='Close', color='black')
ax1.legend()

ax2.plot(df.index, df['DIF'], label='DIF', color='blue')
ax2.plot(df.index, df['DEA'], label='DEA', color='orange')
colors = np.where(df['MACD'] >= 0, 'red', 'green')
ax2.bar(df.index, df['MACD'], color=colors, width=0.7)
ax2.axhline(0, color='gray', lw=0.8)
ax2.legend()
plt.tight_layout()
plt.show()