import numpy as np

def monte_carlo_option_pricing(S, K, T, r, sigma, option_type="call", num_simulations=10000):
    # Generate random variables
    Z = np.random.standard_normal(num_simulations)
    
    # Simulate the stock price at maturity using the geometric Brownian motion
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    # Calculate the payoff for each simulation
    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    elif option_type == "put":
        payoffs = np.maximum(K - ST, 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    # Calculate the average payoff
    average_payoff = np.mean(payoffs)
    
    # Discount the average payoff back to present value
    option_price = np.exp(-r * T) * average_payoff
    
    return option_price


def monte_carlo_with_bootstrapping(S, K, T, r, historical_returns, option_type="call", num_simulations=10000, num_steps=252):
    dt = T / num_steps
    S_paths = np.zeros((num_simulations, num_steps + 1))
    S_paths[:, 0] = S
    
    # Resample returns with replacement for each simulation
    for t in range(1, num_steps + 1):
        sampled_returns = np.random.choice(historical_returns, size=num_simulations, replace=True)
        S_paths[:, t] = S_paths[:, t-1] * np.exp((r - 0.5 * np.var(historical_returns)) * dt + sampled_returns * np.sqrt(dt))
    
    ST = S_paths[:, -1]
    
    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    elif option_type == "put":
        payoffs = np.maximum(K - ST, 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    option_price = np.exp(-r * T) * np.mean(payoffs)
    
    return option_price

# # Example usage:
# S = 100    # Current stock price
# K = 100    # Strike price
# T = 1      # Time to expiration (1 year)
# r = 0.05   # Risk-free rate (5%)
# sigma = 0.2  # Volatility (20%)
# num_simulations = 10000  # Number of simulations

# call_price_mc = monte_carlo_option_pricing(S, K, T, r, sigma, option_type="call", num_simulations=num_simulations)
# put_price_mc = monte_carlo_option_pricing(S, K, T, r, sigma, option_type="put", num_simulations=num_simulations)

# print(f"Call Option Price (Monte Carlo): {call_price_mc:.2f}")
# print(f"Put Option Price (Monte Carlo): {put_price_mc:.2f}")
