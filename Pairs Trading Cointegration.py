import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import statsmodels
import yfinance as yf
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import warnings
warnings.filterwarnings('ignore')


tickers = ['KO','PEP']
df = yf.download(tickers, period='3y', auto_adjust=False)
df = df['Adj Close'].dropna()
df.plot(subplots=True)
plt.show()

score, p_value, _ = coint(df['KO'],df['PEP'])
print("score: ",round(score,4))
print("p-value: ", round(p_value,4))

X = sm.add_constant(df['PEP'])
model = sm.OLS(df['KO'], X).fit()
beta = model.params[1]
print("Estimated hedge ratio (beta):", round(beta, 4))

spread = df['KO'] - beta * df['PEP']
spread.plot(title='KO - beta * PEP (Spread)', figsize=(10,6))
plt.grid(True)
plt.show()


rolling_mean = spread.rolling(window=30).mean()
rolling_std = spread.rolling(window=30).std()
zscore = (spread - rolling_mean) / rolling_std

plt.figure(figsize=(12,6))
plt.plot(zscore, label='Z-score')
plt.axhline(0, color='black', linestyle='--')
plt.axhline(2, color='red', linestyle='--', label='Sell Signal (z>2)')
plt.axhline(-2, color='green', linestyle='--', label='Buy Signal (z<-2)')
plt.legend()
plt.title('Z-Score of Spread (KO vs PEP)')
plt.show()

signal = np.where(zscore > 2, -1, np.nan)
signal = np.where(zscore < -2, 1, signal)
signal = np.where(abs(zscore) < 0.5, 0, signal)

signal = pd.Series(signal, index=zscore.index).ffill().fillna(0)

returns_ko = df['KO'].pct_change()
returns_pep = df['PEP'].pct_change()

spread_ret = returns_ko - beta * returns_pep
strategy_ret = signal.shift(1) * spread_ret

cumulative_ret = (1 + strategy_ret).cumprod() - 1

sharpe_ratio = np.sqrt(252) * (strategy_ret.mean() / strategy_ret.std())

print("Sharpe Ratio:", round(sharpe_ratio, 4))

plt.figure(figsize=(12,6))
plt.plot(cumulative_ret, label='Strategy Cumulative Return')
plt.title('Pairs Trading Strategy Performance')
plt.legend()
plt.show()

cum_ret = (1 + strategy_ret).cumprod()
rolling_max = cum_ret.cummax()
drawdown = (cum_ret - rolling_max) / rolling_max
max_drawdown = drawdown.min()

print("Maximum Drawdown:", round(max_drawdown * 100, 2), "%")

plt.figure(figsize=(12,6))
plt.plot(drawdown, color='red')
plt.title('Drawdown Over Time')
plt.ylabel('Drawdown')
plt.show()
