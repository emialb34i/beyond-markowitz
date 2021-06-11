import random

import numpy as np
import pandas as pd
import yahoo_fin.stock_info as si

from denoise import denoise
from nco import nco
from stock import get_data
from svix import svix2
from nco import markowitz

# 1 - Calculate expected return using SVIX
svix2_stocks = pd.read_csv(
    'svix2_10_6_2021', index_col=0, names=['SVIX2'], header=0)
mkt_cap = pd.read_csv('sp500_mktcap_10_6_2021', index_col=0,
                      names=['mkt_cap'], header=0)
tkrs = list(svix2_stocks.index)

svix_t = svix2('^SPX')
svix_i = svix2_stocks.loc[tkrs]
svix_bar = np.dot(mkt_cap.T, svix2_stocks)/mkt_cap.sum().values
expected_return = (svix_t + 0.5*(svix_i-svix_bar))*2  # annualized

# portfolio will be made of 20 random stocks with expected returns above 10%
tkrs = list(expected_return[expected_return.SVIX2 > .1].index)
tkrs = random.sample(tkrs, 12)

daily_returns = get_data(tkrs).pct_change().dropna()
cov = daily_returns.cov()*252  # annualized
expected_return = expected_return.loc[tkrs]


# 2 - Denoise Covariance Matrix
q = daily_returns.shape[0]/daily_returns.shape[1]  # T/N ratio
cov_denoised = denoise(cov, q)

# 3 - Run NCO algorithm
w = nco(cov_denoised, expected_return.squeeze(), bounds=(0.05, 1))

# 4 - Find the Max Sharpe Ratio allocation
mu = daily_returns.mean(axis=0) * 252  # naive expected return estimate
w_markowitz = markowitz(cov, mu, bounds=(0.05, 1))
w_markowitz = pd.Series(w_markowitz, index=tkrs)
