# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 11:55:37 2026

@author: azz
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


DATA_PATH = r"F:\AXS\20261115\\"

prices = pd.read_parquet(DATA_PATH+ "asx_prices.parquet")

#df = prices[prices.ticker=='DRO.AX']

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
# df = add_macd(df)


def add_macd_signals(df):
    df = add_macd(df)
    # DIF 上穿 DEA = 金叉（1），下穿 = 死叉（-1）
    cross_up   = (df['DIF'] > df['DEA']) & (df['DIF'].shift(1) <= df['DEA'].shift(1))
    cross_down = (df['DIF'] < df['DEA']) & (df['DIF'].shift(1) >= df['DEA'].shift(1))

    df['signal'] = 0
    df.loc[cross_up,   'signal'] = 1
    df.loc[cross_down, 'signal'] = -1
    return df



last2 = sorted(prices['date'].unique())[-2:]              # last two dates in the data
plotted = []                                              # tickers that get a chart

for ticker, df in prices.groupby('ticker'):
    df = add_macd_signals(df)
    cross = df.loc[(df['signal'] == 1) & df['date'].isin(last2), 'date']
    if cross.empty:                                       # no cross up in the last two days
        continue

    lines = []
    for n in (90, 30):                                    # 90-day and 30-day regressions
        reg = df[df['date'] >= df['date'].max() - pd.Timedelta(days=n)]
        x = (reg['date'] - reg['date'].iloc[0]).dt.days
        slope, intercept = np.polyfit(x, reg['close'], 1)
        lines.append((n, reg['date'], intercept + slope * x, slope))
    if min(l[3] for l in lines) <= 0:                     # both slopes must be positive
        continue
    plotted.append(ticker)

    ### Print ###
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})
    ax1.set_title(ticker)
    ax1.plot(df.date, df['close'], label='Close', color='black')
    for (n, d, y, _), c in zip(lines, ('purple', 'teal')):
        ax1.plot(d, y, color=c, ls='--', label=f'{n}-day regression')
    ax1.legend()

    ax2.plot(df.date, df['DIF'], label='DIF (MACD)', color='blue')
    ax2.plot(df.date, df['DEA'], label='DEA', color='orange')
    colors = np.where(df['MACD'] >= 0, 'red', 'green')
    ax2.bar(df.date, df['MACD'], color=colors, width=0.7)
    ax2.axhline(0, color='gray', lw=0.8)
    for d in cross:                                       # gold line at the cross up
        ax1.axvline(d, color='gold', lw=1.5)
        ax2.axvline(d, color='gold', lw=1.5)
    ax2.legend()
    plt.tight_layout()
    fig.savefig(DATA_PATH + ticker + ".png", dpi=150)     # save the charts
    plt.show()

pd.Series(plotted, name='ticker').to_csv(DATA_PATH + "macd_cross_up_tickers.csv", index=False)