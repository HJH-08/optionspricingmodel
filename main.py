import pandas as pd
from data import collect_data
from binomialtree import binomial_tree_option_pricing
from montecarlo import monte_carlo_option_pricing
from blackscholes import black_scholes
import matplotlib.pyplot as plt

FRED_API_KEY = "616c4611ede8eabcf5383214085ea542"

# Define your ticker and date range
ticker_symbol = "AAPL"

# Collect the current data
current_stock_price, current_risk_free_rate, options_data = collect_data(ticker_symbol, FRED_API_KEY)

def backtest_options_pricing(current_stock_price, current_risk_free_rate, options_data, N=100):
    results = []
    for exp_date, options in options_data.items():
        calls = options["calls"]
        for index, option in calls.iterrows():
            # Use today's date as the start date for all calculations
            date = pd.Timestamp.today()

            S = current_stock_price
            r = current_risk_free_rate
            T = (pd.to_datetime(exp_date) - date).days / 365  # Time to expiration from today
            K = option['strike']
            sigma = option['impliedVolatility']

            # Filter out or adjust low sigma values
            if sigma < 0.01:
                continue

            # Calculate prices using all three models
            binomial_price = binomial_tree_option_pricing(S, K, T, r, sigma, N, option_type="call", american=True)
            monte_carlo_price = monte_carlo_option_pricing(S, K, T, r, sigma, option_type="call")
            black_scholes_price = black_scholes(S, K, T, r, sigma, option_type="call")
            
            # Compare with market price and record the results
            market_price = option['lastPrice']
            results.append({
                'Date': date,
                'Strike': K,
                'Expiration': exp_date,
                'MarketPrice': market_price,
                'BinomialPrice': binomial_price,
                'MonteCarloPrice': monte_carlo_price,
                'BlackScholesPrice': black_scholes_price,
                'BinomialError': market_price - binomial_price,
                'MonteCarloError': market_price - monte_carlo_price,
                'BlackScholesError': market_price - black_scholes_price,
            })
    
    results_df = pd.DataFrame(results)
    return results_df




# Perform the backtest
backtest_results = backtest_options_pricing(current_stock_price, current_risk_free_rate, options_data)
print(backtest_results.head())
# # Display the results
# print(backtest_results)

# Sort the DataFrame by the absolute value of the 'Error' column in descending order
#sorted_results_df = backtest_results.reindex(backtest_results['Error'].abs().sort_values(ascending=False).index)

# # Display the top rows sorted by error magnitude
# print(sorted_results_df.head())


# Analyze the results
# mean_error = backtest_results['Error'].mean()
# print(f"Mean Error: {mean_error:.2f}")




# # Plot theoretical vs market prices
# plt.figure(figsize=(10, 6))
# plt.plot(backtest_results['Date'], backtest_results['MarketPrice'], label='Market Price', marker='o')
# plt.plot(backtest_results['Date'], backtest_results['TheoreticalPrice'], label='Theoretical Price', marker='x')
# plt.xlabel('Date')
# plt.ylabel('Option Price')
# plt.title('Theoretical vs Market Prices')
# plt.legend()
# plt.show()

# # Plot errors
# plt.figure(figsize=(10, 6))
# plt.plot(backtest_results['Date'], backtest_results['Error'], label='Pricing Error', marker='o')
# plt.xlabel('Date')
# plt.ylabel('Error')
# plt.title('Pricing Error Over Time')
# plt.legend()
# plt.show()
