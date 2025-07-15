import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Get NIFTY 50 data for past 1 year
data = yf.download("^NSEI", period="1y")
close_prices = data['Close'].dropna()

# Step 2: Calculate daily returns
returns = close_prices.pct_change().dropna()
mu = returns.mean()
sigma = returns.std()

# Step 3: Set simulation parameters
S0 = close_prices.iloc[-1]   # Last known NIFTY price
days = 30                    # How many days to simulate
simulations = 10000           # How many paths to simulate

# Step 4: Monte Carlo simulation
results = np.zeros((days, simulations))

for sim in range(simulations):
    price_path = np.zeros(days)
    price_path[0] = S0
    for day in range(1, days):
        random_shock = np.random.normal(loc=0.0, scale=1.0)
        price_path[day] = price_path[day-1] * np.exp((mu - 0.5 * sigma**2) + sigma * random_shock)
    results[:, sim] = price_path

# Step 5: Plot simulation paths
plt.figure(figsize=(12,6))
plt.plot(results, color='grey', alpha=0.1)
plt.title("Monte Carlo Simulation of NIFTY 50 (Next 30 Days)")
plt.xlabel("Days Ahead")
plt.ylabel("Simulated NIFTY 50 Price")
plt.grid()
plt.show()

# Step 6: Show prediction range on last day
final_prices = results[-1, :]
print(f"Predicted NIFTY 50 range after {days} days:")
print(f"Minimum: ₹{round(final_prices.min(), 2)}")
print(f"Maximum: ₹{round(final_prices.max(), 2)}")
print(f"Average: ₹{round(final_prices.mean(), 2)}")
print(f"Std Dev: ₹{round(final_prices.std(), 2)}")
