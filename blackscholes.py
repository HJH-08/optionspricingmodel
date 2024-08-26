import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d
from fredapi import Fred

FRED_API_KEY = "616c4611ede8eabcf5383214085ea542"
fred = Fred(api_key=FRED_API_KEY)

def get_risk_free_rate(T, rate_curve):
    # Extract maturities and rates
    times, rates = zip(*rate_curve.items())
    
    # Create an interpolation function
    rate_interpolator = interp1d(times, rates, fill_value="extrapolate")
    
    # Return the interpolated rate for the given time to maturity
    return rate_interpolator(T)

def get_rate_curve():
    # Define the maturity series you want from FRED
    maturity_series = {
        1.0: 'DGS1',  # 1-Year Treasury
        2.0: 'DGS2',  # 2-Year Treasury
        5.0: 'DGS5',  # 5-Year Treasury
        10.0: 'DGS10'  # 10-Year Treasury
    }
    
    rate_curve = {}
    for maturity, series_id in maturity_series.items():
        # Fetch the latest available rate using the instance 'fred'
        rate_series = fred.get_series(series_id).dropna()
        latest_rate = rate_series.iloc[-1] / 100  # Convert to decimal
        rate_curve[maturity] = latest_rate
    
    return rate_curve

rate_curve = get_rate_curve()

def black_scholes_rate_curved(S, K, T, rate_curve, sigma, option_type="call"):
    # Get the appropriate risk-free rate for the given maturity
    r = get_risk_free_rate(T, rate_curve)
    
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate option price
    if option_type == "call":
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    return option_price

def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    return option_price

# S = 100  # Current stock price
# K = 100  # Strike price
# T = 1.5  # Time to expiration (1.5 years) 
# sigma = 0.2  # Volatility (20%)

# call_price = black_scholes(S, K, T, rate_curve, sigma, option_type="call")
# put_price = black_scholes(S, K, T, rate_curve, sigma, option_type="put")

# print(f"Call Option Price: {call_price:.2f}")
# print(f"Put Option Price: {put_price:.2f}")
