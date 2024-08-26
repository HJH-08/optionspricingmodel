import pandas as pd
from data import collect_data, get_historical_returns
from binomialtree import binomial_tree_option_pricing
from montecarlo import monte_carlo_option_pricing
from blackscholes import black_scholes, black_scholes_rate_curved, get_rate_curve, get_risk_free_rate
from montecarlo import monte_carlo_with_bootstrapping  # Import the new function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FRED_API_KEY = "616c4611ede8eabcf5383214085ea542"

# Define your ticker and date range
ticker_symbol = "AAPL"
start_date = "2023-01-01"
end_date = "2024-01-01"

# Get historical returns
historical_returns = get_historical_returns(ticker_symbol, start_date, end_date)
# Collect the current data
current_stock_price, current_risk_free_rate, options_data = collect_data(ticker_symbol, FRED_API_KEY)

# Get the rate curve for Black-Scholes with rate curve
rate_curve = get_rate_curve()
for T in [0.5, 1, 2, 3, 5, 10]:  # Example maturities
    interpolated_rate = get_risk_free_rate(T, rate_curve)
    print(f"Interpolated rate for {T} years: {interpolated_rate:.4f}")

def backtest_options_pricing(current_stock_price, current_risk_free_rate, options_data, rate_curve, historical_returns, N=100, otm_threshold=0.2):
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

            # Filter out very OTM options
            if K > S * (1 + otm_threshold):
                continue

            # Filter out or adjust low sigma values
            if sigma < 0.01:
                continue

            # Calculate prices using all five models
            binomial_price = binomial_tree_option_pricing(S, K, T, r, sigma, N, option_type="call", american=True)
            monte_carlo_price = monte_carlo_option_pricing(S, K, T, r, sigma, option_type="call")
            black_scholes_price = black_scholes(S, K, T, r, sigma, option_type="call")
            black_scholes_curve_price = black_scholes_rate_curved(S, K, T, rate_curve, sigma, option_type="call")
            monte_carlo_bootstrapping_price = monte_carlo_with_bootstrapping(S, K, T, r, historical_returns, option_type="call", num_simulations=10000, num_steps=252)
            
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
                'BlackScholesCurvePrice': black_scholes_curve_price,
                'MonteCarloBootstrappingPrice': monte_carlo_bootstrapping_price,
                'BinomialError': market_price - binomial_price,
                'MonteCarloError': market_price - monte_carlo_price,
                'BlackScholesError': market_price - black_scholes_price,
                'BlackScholesCurveError': market_price - black_scholes_curve_price,
                'MonteCarloBootstrappingError': market_price - monte_carlo_bootstrapping_price
            })
    
    results_df = pd.DataFrame(results)
    return results_df


def evaluate_models(backtest_results):
    # Calculate the evaluation metrics for each model
    metrics = {
        'Model': [],
        'MAE': [],
        'MSE': [],
        'RMSE': [],
        'R²': []
    }
    
    models = ['Binomial', 'MonteCarlo', 'BlackScholes', 'BlackScholesCurve', 'MonteCarloBootstrapping']
    
    for model in models:
        mae = mean_absolute_error(backtest_results['MarketPrice'], backtest_results[f'{model}Price'])
        mse = mean_squared_error(backtest_results['MarketPrice'], backtest_results[f'{model}Price'])
        rmse = np.sqrt(mse)
        r2 = r2_score(backtest_results['MarketPrice'], backtest_results[f'{model}Price'])
        
        metrics['Model'].append(model)
        metrics['MAE'].append(mae)
        metrics['MSE'].append(mse)
        metrics['RMSE'].append(rmse)
        metrics['R²'].append(r2)
    
    metrics_df = pd.DataFrame(metrics)
    return metrics_df

# Perform the backtest with OTM filtering
backtest_results = backtest_options_pricing(current_stock_price, current_risk_free_rate, options_data, rate_curve, historical_returns, otm_threshold=0.2)

print(backtest_results)
# Evaluate the models
metrics_df = evaluate_models(backtest_results)
print(metrics_df)

# Visual comparison
plt.figure(figsize=(14, 8))
plt.scatter(backtest_results['MarketPrice'], backtest_results['BinomialPrice'], label='Binomial', alpha=0.5)
plt.scatter(backtest_results['MarketPrice'], backtest_results['MonteCarloPrice'], label='Monte Carlo', alpha=0.5)
plt.scatter(backtest_results['MarketPrice'], backtest_results['BlackScholesPrice'], label='Black-Scholes', alpha=0.5)
plt.scatter(backtest_results['MarketPrice'], backtest_results['BlackScholesCurvePrice'], label='Black-Scholes Curve', alpha=0.5)
plt.scatter(backtest_results['MarketPrice'], backtest_results['MonteCarloBootstrappingPrice'], label='Monte Carlo Bootstrapping', alpha=0.5)
plt.plot([min(backtest_results['MarketPrice']), max(backtest_results['MarketPrice'])],
         [min(backtest_results['MarketPrice']), max(backtest_results['MarketPrice'])],
         color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel('Market Price')
plt.ylabel('Predicted Price')
plt.title('Market Price vs Predicted Price')
plt.legend()
plt.show()

# Residual plots (errors over time)
plt.figure(figsize=(14, 8))
plt.plot(backtest_results['Date'], backtest_results['BinomialError'], label='Binomial Error', marker='o')
plt.plot(backtest_results['Date'], backtest_results['MonteCarloError'], label='Monte Carlo Error', marker='x')
plt.plot(backtest_results['Date'], backtest_results['BlackScholesError'], label='Black-Scholes Error', marker='^')
plt.plot(backtest_results['Date'], backtest_results['BlackScholesCurveError'], label='Black-Scholes Curve Error', marker='s')
plt.plot(backtest_results['Date'], backtest_results['MonteCarloBootstrappingError'], label='Monte Carlo Bootstrapping Error', marker='d')
plt.xlabel('Date')
plt.ylabel('Error')
plt.title('Pricing Errors Over Time')
plt.legend()
plt.show()
