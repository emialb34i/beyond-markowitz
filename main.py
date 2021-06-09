from stock import get_data
import yahoo_fin.stock_info as si
from denoise import denoise
import random

random.seed(23)

tkrs = random.sample(si.tickers_sp500(), 20)
daily_returns = get_data(tkrs).pct_change().dropna()
cov = daily_returns.cov()*252  # annualized

# 1 - Calculate expected return using SVIX

# 2 - Denoise Covariance Matrix
q = daily_returns.shape[0]/daily_returns.shape[1]  # T/N ratio
cov_denoised = denoise(cov, q)

# 3 - Run NCO algorithm
