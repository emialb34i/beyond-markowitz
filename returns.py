from datetime import datetime

import numpy as np
import pandas as pd
import yahoo_fin.stock_info as si
import yfinance as yf
from scipy.stats import zscore

from svix import svix2

# list of all SP500 tickers
sp500_tickers = np.array(si.tickers_sp500())


def svix2_bullet(ticker, month_to_expiry):
    try:
        return svix2(ticker, month_to_expiry)
    except:
        # print(f"{ticker} - could not calc. SVIX2")
        return -1


# Calcultes SVIX2 for each stock in the SP500
svix2_sp500 = np.zeros(len(sp500_tickers))

for i in range(len(svix2_sp500)):
    if i % 10 == 0:
        print(i)
    tkr = sp500_tickers[i]
    val = svix2_bullet(tkr, 6)
    # if val > 0:
    #     print(tkr, val)
    svix2_sp500[i] = val


# Convert to a series at is easier to work with
svix2_sp500 = pd.Series(svix2_sp500, index=sp500_tickers)

# Remove tickers for which we couldnt calc SVIX2 - (usually means 6 month options don't exist)
sp500_tickers = sp500_tickers[svix2_sp500 > 0]
svix2_sp500 = svix2_sp500[sp500_tickers]

# Remove Outliers
z_scores = zscore(svix2_sp500)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 2.5)
svix2_sp500 = svix2_sp500[filtered_entries]
sp500_tickers = svix2_sp500.index

# save svix2 as csv
filename = f"svix2_{datetime.now().day}_{datetime.now().month}_{datetime.now().year}.csv"
svix2_sp500.to_csv(filename)

# Fetch market cap for each stock left
sp500_mktcap = np.zeros(len(svix2_sp500))
for i in range(len(svix2_sp500.index)):
    if i % 10 == 0:
        print(i)
    tkr = svix2_sp500.index[i]
    mktcap = yf.Ticker(tkr).info['marketCap']
    sp500_mktcap[i] = mktcap
    # print(tkr, sp500_mktcap[i])

sp500_mktcap = np.array(sp500_mktcap)

# save mktcap as CSV
filename = f"sp500_mktcap_{datetime.now().day}_{datetime.now().month}_{datetime.now().year}.csv"
pd.Series(sp500_mktcap, index=svix2_sp500.index).to_csv(filename)
