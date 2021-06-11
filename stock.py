import random

import numpy as np
import pandas as pd
import yahoo_fin.stock_info as si
import yfinance as yf


def get_data(tkrs):
    data = yf.download(tkrs, period="1y", group_by='ticker')
    df = pd.DataFrame()
    for tkr in tkrs:
        df[tkr] = data[tkr]['Adj Close']
    df.fillna(method='bfill', inplace=True)
    return df
