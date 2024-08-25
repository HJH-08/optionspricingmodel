import numpy as np
import matplotlib.pyplot as plt

def binomial_tree_option_pricing(S, K, T, r, sigma, N, option_type="call", american=False):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    q = (np.exp(r * dt) - d) / (u - d)
    discount_factor = np.exp(-r * dt)
    
    stock_prices = np.zeros((N + 1, N + 1))
    option_values = np.zeros((N + 1, N + 1))
    
    for i in range(N + 1):
        for j in range(i + 1):
            stock_prices[j, i] = S * (u ** (i - j)) * (d ** j)
    
    if option_type == "call":
        option_values[:, N] = np.maximum(stock_prices[:, N] - K, 0)
    elif option_type == "put":
        option_values[:, N] = np.maximum(K - stock_prices[:, N], 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option_values[j, i] = (q * option_values[j, i + 1] + (1 - q) * option_values[j + 1, i + 1]) * discount_factor
            if american:
                if option_type == "call":
                    option_values[j, i] = np.maximum(option_values[j, i], stock_prices[j, i] - K)
                elif option_type == "put":
                    option_values[j, i] = np.maximum(option_values[j, i], K - stock_prices[j, i])
    
    return option_values[0, 0]


def calculate_greeks(S, K, T, r, sigma, N, option_type="call", american=False):
    dS = 1e-4 * S  # Small change in stock price
    dT = 1e-5 * T  # Small change in time
    dr = 1e-5  # Small change in risk-free rate
    dSigma = 1e-5  # Small change in volatility
    
    # Calculate option prices for different scenarios
    price = binomial_tree_option_pricing(S, K, T, r, sigma, N, option_type, american)
    price_up = binomial_tree_option_pricing(S + dS, K, T, r, sigma, N, option_type, american)
    price_down = binomial_tree_option_pricing(S - dS, K, T, r, sigma, N, option_type, american)
    price_T_down = binomial_tree_option_pricing(S, K, T - dT, r, sigma, N, option_type, american)
    price_r_up = binomial_tree_option_pricing(S, K, T, r + dr, sigma, N, option_type, american)
    price_sigma_up = binomial_tree_option_pricing(S, K, T, r, sigma + dSigma, N, option_type, american)
    
    # Delta
    delta = (price_up - price_down) / (2 * dS)
    
    # Gamma
    gamma = (price_up - 2 * price + price_down) / (dS ** 2)
    
    # Theta
    theta = (price_T_down - price) / dT
    
    # Vega
    vega = (price_sigma_up - price) / dSigma
    
    # Rho
    rho = (price_r_up - price) / dr
    
    return {"Delta": delta, "Gamma": gamma, "Theta": theta, "Vega": vega, "Rho": rho}

# Example usage:
S = 100    # Current stock price
K = 100    # Strike price
T = 1      # Time to expiration (1 year)
r = 0.05   # Risk-free rate (5%)
sigma = 0.2  # Volatility (20%)
N = 100    # Number of time steps

# greeks = calculate_greeks(S, K, T, r, sigma, N, option_type="call", american=True)


def plot_greeks(S, K, T, r, sigma, N, option_type="call", american=False):
    S_range = np.linspace(S * 0.8, S * 1.2, 100)
    delta_values = []
    gamma_values = []
    
    for s in S_range:
        greeks = calculate_greeks(s, K, T, r, sigma, N, option_type, american)
        delta_values.append(greeks["Delta"])
        gamma_values.append(greeks["Gamma"])
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(S_range, delta_values, label='Delta')
    plt.title('Delta vs. Stock Price')
    plt.xlabel('Stock Price')
    plt.ylabel('Delta')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(S_range, gamma_values, label='Gamma', color='orange')
    plt.title('Gamma vs. Stock Price')
    plt.xlabel('Stock Price')
    plt.ylabel('Gamma')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

#plot_greeks(S, K, T, r, sigma, N, option_type="call", american=True)

