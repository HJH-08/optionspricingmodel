import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d

def get_risk_free_rate(T, rate_curve):
    # Extract maturities and rates
    times, rates = zip(*rate_curve.items())
    
    # Create an interpolation function
    rate_interpolator = interp1d(times, rates, fill_value="extrapolate")
    
    # Return the interpolated rate for the given time to maturity
    return rate_interpolator(T)

rate_curve = {
    0.5: 0.03,  # 6-month interest rate is 3%
    1.0: 0.05,  # 1-year interest rate is 5%
    2.0: 0.06,  # 2-year interest rate is 6%
    5.0: 0.07   # 5-year interest rate is 7%
}


def black_scholes(S, K, T, rate_curve, sigma, option_type="call"):
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


S = 100  # Current stock price
K = 100  # Strike price
T = 1.5  # Time to expiration (1.5 years) 
sigma = 0.2  # Volatility (20%)

call_price = black_scholes(S, K, T, rate_curve, sigma, option_type="call")
put_price = black_scholes(S, K, T, rate_curve, sigma, option_type="put")

print(f"Call Option Price: {call_price:.2f}")
print(f"Put Option Price: {put_price:.2f}")
