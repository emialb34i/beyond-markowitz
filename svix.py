from scipy.stats import zscore
import yahoo_fin.stock_info as si
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.integrate import simps

# RUN THIS WHEN MARKET IS CLOSED


def month_fix(x):
    current_year = datetime.now().year
    current_month = datetime.now().month
    multiplier = x.year - current_year
    return (x.month-current_month) + multiplier*12


def get_options_months(ticker):
    return pd.to_datetime(pd.Series(ticker.options)).apply(lambda x: month_fix(x))


def get_options_date(ticker, time_to_expiry=6):
    # given a time frame to expiry in months it finds the closest option to the expiry month
    # ex - you want 7 month options for a stock, if 7 months options exist it returns the expiry date,
    # else it returns the closest options it finds
    idx = find_nearest_index(get_options_months(ticker), time_to_expiry)
    return ticker.options[idx]


def shrink_options(df, strike_min=None, strike_max=None):
    if strike_min and strike_max:
        return df.loc[(strike_min <= df['strike']) & (df['strike'] <= strike_max)]
    elif strike_min:
        return df.loc[strike_min <= df['strike']]
    else:
        return df.loc[df['strike'] <= strike_max]


def fix_strike_prices(strike_call, strike_put):
    # finds common strike prices from call and put strike prices
    strike = list(set(strike_call) & set(strike_put))
    return pd.Series(strike).sort_values()


def get_options(ticker, date):
    options = ticker.option_chain(date=date)
    call = options.calls[['lastPrice', 'strike']]
    put = options.puts[['lastPrice', 'strike']]
    return call, put


def get_current_price(ticker):
    if type(ticker) == str:
        ticker = yf.Ticker(ticker)
    todays_data = ticker.history(period='1d')
    return todays_data['Close'][0]


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def svix2(ticker, month_to_expiry=6):
    ticker = yf.Ticker(ticker)

    # finds closest option date
    # ex. - if you want a 6 month option it finds the date of closest 6 month option
    date = get_options_date(ticker, month_to_expiry)

    # get option dataframe price
    call_df, put_df = get_options(ticker, date)

    price = get_current_price(ticker)
    risk_free = 1 + 0.04/100  # 6-month risk free rate

    # This is the F in the integral bound, since it has to be a strike value
    # it finds the closest strike to the price and sets that as the bound
    strike = fix_strike_prices(call_df['strike'], put_df['strike'])
    # small fix, sometimes calls and put dont have the same strike prices
    # this creates an array with all the common put and call strikes
    integration_bound = find_nearest(strike, price)

    # sets up the call and put for integration, I select all the calls with strike greater
    # than F and all the puts less than F
    call = shrink_options(call_df, strike_min=integration_bound)
    put = shrink_options(put_df, strike_max=integration_bound)

    # numerical integration of call and put
    call_integral = simps(call['lastPrice'], call['strike'])
    put_integral = simps(put['lastPrice'], put['strike'])

    integrals = call_integral+put_integral
    first_term = 2/(risk_free*(price**2))

    # compute the SVIX^2
    svix2 = first_term*integrals

    return svix2
