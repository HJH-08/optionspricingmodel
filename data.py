import yfinance as yf
import pandas as pd
from fredapi import Fred

def collect_data(ticker_symbol, api_key):
    # Download current stock price
    stock_data = yf.download(ticker_symbol, period="1d")  # Fetches the latest data

    # Get the most recent closing price
    current_stock_price = stock_data['Close'].iloc[-1]

    # Fetch the current 10-Year Treasury Constant Maturity Rate (DGS10)
    fred = Fred(api_key=api_key)
    current_risk_free_rate = fred.get_series('DGS10').iloc[-1] / 100  # Get the latest value and convert to decimal

    # Get the ticker object and expiration dates for options data
    ticker = yf.Ticker(ticker_symbol)
    expirations = ticker.options

    # Create a dictionary to hold options data
    options_data = {}

    # Loop through expiration dates to get options data
    for exp_date in expirations:
        options_chain = ticker.option_chain(exp_date)
        options_data[exp_date] = {
            "calls": options_chain.calls,
            "puts": options_chain.puts
        }

    return current_stock_price, current_risk_free_rate, options_data
