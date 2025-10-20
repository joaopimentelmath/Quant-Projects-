import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.polynomial.polynomial import polyvander
import time
import os

# ---------- Settings ----------
np.random.seed(2025)           # reproducible results
out_dir = "quant_outputs"
os.makedirs(out_dir, exist_ok=True)

# Fake historical generation params
S0_hist = 100.0                # initial price for synthetic history
mu_true = 0.06                 # true annual drift used to create fake history
sigma_true = 0.22              # true annual volatility used to create fake history
history_days = 252 * 3         # 3 years of trading days
dt_hist = 1.0 / 252

# Simulation / pricing params
r = 0.03                       # risk-free rate used in pricing (risk-neutral drift)
T = 1.0                        # option maturity in years
steps = 252                    # discretization steps per path
n_paths_default = 20000        # default Monte Carlo sample size, adjust if needed
K = 100.0                      # strike price for options, can be changed
use_antithetic = True          # use antithetic variates for variance reduction

# Finite difference bump sizes for Greeks
eps_rel_S = 1e-2               # relative bump for S0, 1%
eps_sigma = 1e-4
eps_r = 1e-4
eps_T = 1.0 / 252.0            # one trading day in years


# ---------- Utilities ----------
def generate_fake_history(S0, mu, sigma, days, dt):
    """
    Generate synthetic historical adjusted close prices via GBM.
    Returns a pandas Series indexed 0..days
    """
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal(size=days)
    log_prices = np.concatenate([[0.0], np.cumsum(increments)])
    prices = S0 * np.exp(log_prices)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days + 1, freq='B')  # business days
    return pd.Series(prices, index=dates, name="Close")


def estimate_mu_sigma_from_returns(price_series, trading_days_per_year=252):
    log_returns = np.log(price_series / price_series.shift(1)).dropna()
    mu_hat = log_returns.mean() * trading_days_per_year
    sigma_hat = log_returns.std(ddof=1) * np.sqrt(trading_days_per_year)
    return mu_hat, sigma_hat, log_returns


def simulate_gbm_paths(S0, drift, sigma, T, steps, n_paths, antithetic=False, Z=None):
    dt = T / steps
    if Z is None:
        Z = np.random.normal(size=(n_paths, steps))
        if antithetic:
            # half randoms, then mirror them for variance reduction
            half = n_paths // 2
            Z[:half, :] = np.random.normal(size=(half, steps))
            Z[half:2*half, :] = -Z[:half, :]
            if n_paths % 2 == 1:
                Z[-1, :] = np.random.normal(size=(steps,))
    increments = (drift - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)], axis=1)
    S_paths = S0 * np.exp(log_paths)
    return S_paths, Z


def mc_european_price(S0, r, sigma, T, steps, n_paths, K, option_type='call', antithetic=False, Z=None):
    drift = r  # risk-neutral
    S_paths, used_Z = simulate_gbm_paths(S0, drift, sigma, T, steps, n_paths, antithetic, Z)
    ST = S_paths[:, -1]
    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0.0)
    else:
        payoffs = np.maximum(K - ST, 0.0)
    price = np.exp(-r * T) * np.mean(payoffs)
    se = np.exp(-r * T) * np.std(payoffs, ddof=1) / np.sqrt(n_paths)
    return price, se, used_Z, ST


def bs_price_call_put(S0, K, r, sigma, T):
    if sigma <= 0 or T <= 0:
        call = max(S0 - K, 0.0)
        put = max(K - S0, 0.0)
        return call, put
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return call, put


def bs_call_greeks(S0, K, r, sigma, T):
    if sigma <= 0 or T <= 0:
        return {"price": max(S0 - K, 0.0), "delta": np.nan, "gamma": np.nan, "vega": np.nan, "theta": np.nan, "rho": np.nan}
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    vega = S0 * norm.pdf(d1) * np.sqrt(T)
    theta = (-S0 * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2))
    rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    return {"price": price, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


# Least Squares Monte Carlo for American option pricing, using polynomial basis
def lsm_american_put_price(S0, r, sigma, T, steps, n_paths, K, poly_degree=2, antithetic=False):
    S_paths, _ = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, antithetic)
    dt = T / steps
    discount = np.exp(-r * dt)
    # payoff matrix: shape (n_paths, steps+1)
    payoffs = np.maximum(K - S_paths, 0.0)
    cashflows = payoffs[:, -1].copy()
    # backward induction
    for t in range(steps-1, 0, -1):
        itm = payoffs[:, t] > 0
        if not np.any(itm):
            cashflows *= discount
            continue
        X = S_paths[itm, t]
        Y = cashflows[itm] * discount
        # regress Y on polynomial basis using numpy polyvander with increasing powers
        V = polyvander(X, poly_degree)  # columns pow 0..degree
        coeffs, *_ = np.linalg.lstsq(V, Y, rcond=None)
        continuation = V.dot(coeffs)
        exercise = payoffs[itm, t]
        exercise_now = exercise > continuation
        # update cashflows: for exercised paths set to exercise value, for others keep continuation (implicitly)
        exercised_indices = np.where(itm)[0][exercise_now]
        cashflows[exercised_indices] = exercise[exercise_now]
        cashflows *= discount
    price = np.mean(cashflows)
    return price


# Pathwise Delta estimator for European call. For GBM this is exact under continuous model
def pathwise_delta_estimator(S_paths, K, r, T):
    # For a call, payoff = max(ST - K, 0), pathwise derivative wrt S0 is
    # if ST > K then derivative = ST / S0 * exp(-rT), else 0. So Delta_pw = exp(-rT) * 1_{ST>K} * (ST/S0)
    ST = S_paths[:, -1]
    indicator = (ST > K).astype(float)
    delta_pw = np.exp(-r * T) * indicator * (ST / S_paths[0, 0])  # assumes S_paths[0,0] = S0
    return delta_pw.mean()


# ---------- Main flow ----------

def main():
    # 1. Generate fake historical series
    hist_prices = generate_fake_history(S0_hist, mu_true, sigma_true, history_days, dt_hist)
    mu_hat, sigma_hat, log_returns = estimate_mu_sigma_from_returns(hist_prices)
    S0_est = hist_prices.iloc[-1]

    print("===== Synthetic historical summary =====")
    print(f"initial S0_hist: {S0_hist:.2f}, last price S0_est: {S0_est:.2f}")
    print(f"true mu used to generate history: {mu_true:.4f}, true sigma: {sigma_true:.4f}")
    print(f"estimated mu from history: {mu_hat:.4f}, estimated sigma from history: {sigma_hat:.4f}")
    print("========================================\n")

    # 2. Simulate GBM under risk-neutral measure and plot sample paths
    n_plot_paths = 20
    sim_S, _ = simulate_gbm_paths(S0_est, r, sigma_hat, T, steps, n_plot_paths, antithetic=use_antithetic)
    times = np.linspace(0, T, steps + 1)

    plt.figure(figsize=(10, 6))
    for i in range(n_plot_paths):
        plt.plot(times, sim_S[i, :], lw=1)
    plt.title("Sample simulated GBM paths, synthetic stock")
    plt.xlabel("Time (years)")
    plt.ylabel("Price")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sample_paths.png"))
    print(f"Saved sample GBM paths to {out_dir}/sample_paths.png")

    # 3. Monte Carlo price for European call and put
    n_paths = n_paths_default
    start_time = time.time()
    price_call_mc, se_call_mc, Z_used, STs = mc_european_price(S0_est, r, sigma_hat, T, steps, n_paths, K, option_type='call', antithetic=use_antithetic)
    price_put_mc, se_put_mc, _, _ = mc_european_price(S0_est, r, sigma_hat, T, steps, n_paths, K, option_type='put', antithetic=use_antithetic)
    elapsed = time.time() - start_time
    print("\nMonte Carlo pricing results")
    print(f"European call MC price: {price_call_mc:.6f}, se: {se_call_mc:.6f}, time: {elapsed:.2f}s")
    print(f"European put  MC price: {price_put_mc:.6f}, se: {se_put_mc:.6f}")

    # 4. Black-Scholes analytic comparison
    bs_call, bs_put = bs_price_call_put(S0_est, K, r, sigma_hat, T)
    bs_greeks = bs_call_greeks(S0_est, K, r, sigma_hat, T)
    print("\nBlack-Scholes comparison")
    print(f"BS call price: {bs_call:.6f}, BS put price: {bs_put:.6f}")
    print("BS Greeks:", bs_greeks)

    # 5. LSM American put price
    print("\nComputing LSM American put price, this may take a few seconds...")
    lsm_price = lsm_american_put_price(S0_est, r, sigma_hat, T, steps, n_paths//5, K, poly_degree=2, antithetic=use_antithetic)
    print(f"LSM American put price (approx): {lsm_price:.6f}")

    # 6. Greeks via finite differences using common random numbers
    print("\nComputing numerical Greeks via central finite differences with common random numbers...")
    # reuse Z_used for common randomness if antithetic variant created Z_used shape as (n_paths, steps)
    if Z_used is None or Z_used.shape[0] < n_paths:
        # generate fresh common Z
        Z_common = np.random.normal(size=(n_paths, steps))
    else:
        Z_common = Z_used

    eps_S = eps_rel_S * S0_est
    # price baseline
    price_base, se_base, _, _ = mc_european_price(S0_est, r, sigma_hat, T, steps, n_paths, K, option_type='call', antithetic=use_antithetic, Z=Z_common)

    # Delta and Gamma
    price_up_S, _, _, _ = mc_european_price(S0_est + eps_S, r, sigma_hat, T, steps, n_paths, K, option_type='call', antithetic=use_antithetic, Z=Z_common)
    price_down_S, _, _, _ = mc_european_price(S0_est - eps_S, r, sigma_hat, T, steps, n_paths, K, option_type='call', antithetic=use_antithetic, Z=Z_common)
    delta_mc = (price_up_S - price_down_S) / (2 * eps_S)
    gamma_mc = (price_up_S - 2 * price_base + price_down_S) / (eps_S**2)

    # Vega
    price_up_sig, _, _, _ = mc_european_price(S0_est, r, sigma_hat + eps_sigma, T, steps, n_paths, K, option_type='call', antithetic=use_antithetic, Z=Z_common)
    price_down_sig, _, _, _ = mc_european_price(S0_est, r, sigma_hat - eps_sigma, T, steps, n_paths, K, option_type='call', antithetic=use_antithetic, Z=Z_common)
    vega_mc = (price_up_sig - price_down_sig) / (2 * eps_sigma)

    # Rho
    price_up_r, _, _, _ = mc_european_price(S0_est, r + eps_r, sigma_hat, T, steps, n_paths, K, option_type='call', antithetic=use_antithetic, Z=Z_common)
    price_down_r, _, _, _ = mc_european_price(S0_est, r - eps_r, sigma_hat, T, steps, n_paths, K, option_type='call', antithetic=use_antithetic, Z=Z_common)
    rho_mc = (price_up_r - price_down_r) / (2 * eps_r)

    # Theta
    if T - eps_T <= 0:
        theta_mc = np.nan
    else:
        steps_short = max(1, int(round(steps * (T - eps_T) / T)))
        price_short, _, _, _ = mc_european_price(S0_est, r, sigma_hat, T - eps_T, steps_short, n_paths, K, option_type='call', antithetic=use_antithetic)
        theta_mc = (price_short - price_base) / (-eps_T)

    # Pathwise Delta estimator (low-noise)
    S_paths_pw, _ = simulate_gbm_paths(S0_est, r, sigma_hat, T, steps, n_paths, antithetic=use_antithetic, Z=Z_common)
    delta_pw = pathwise_delta_estimator(S_paths_pw, K, r, T)

    print("\nNumerical Greeks (Monte Carlo finite differences):")
    print(f"Delta (MC FD)  : {delta_mc:.6f}")
    print(f"Gamma (MC FD)  : {gamma_mc:.6f}")
    print(f"Vega  (MC FD)  : {vega_mc:.6f}")
    print(f"Rho   (MC FD)  : {rho_mc:.6f}")
    print(f"Theta (MC FD)  : {theta_mc:.6f}")
    print(f"Delta (pathwise): {delta_pw:.6f}")

    print("\nBlack-Scholes Greeks for comparison:")
    print(f"Delta (BS) = {bs_greeks['delta']:.6f}, Gamma (BS) = {bs_greeks['gamma']:.6f}, Vega (BS) = {bs_greeks['vega']:.6f}")
    print(f"Rho (BS) = {bs_greeks['rho']:.6f}, Theta (BS) = {bs_greeks['theta']:.6f}")

    # 7. Plots: histogram of terminal call payoffs and convergence plot
    payoffs = np.exp(-r * T) * np.maximum(STs - K, 0.0)
    plt.figure(figsize=(8,5))
    plt.hist(payoffs, bins=60, edgecolor='k', alpha=0.6)
    plt.title("Discounted terminal call payoffs histogram")
    plt.xlabel("Discounted payoff")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "payoff_histogram.png"))
    print(f"Saved payoff histogram to {out_dir}/payoff_histogram.png")

    # Convergence of MC price vs number of paths
    Ns = np.unique(np.round(np.logspace(2, np.log10(n_paths), num=8)).astype(int))
    conv_prices = []
    conv_ses = []
    for N in Ns:
        p, se, _, _ = mc_european_price(S0_est, r, sigma_hat, T, steps, N, K, option_type='call', antithetic=use_antithetic)
        conv_prices.append(p)
        conv_ses.append(se)
    plt.figure(figsize=(8,5))
    plt.errorbar(Ns, conv_prices, yerr=conv_ses, fmt='-o')
    plt.xscale('log')
    plt.title("Convergence of Monte Carlo Call Price vs Number of Paths")
    plt.xlabel("Number of paths (log scale)")
    plt.ylabel("Call price ± SE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "convergence.png"))
    print(f"Saved convergence plot to {out_dir}/convergence.png")

    # 8. Save numeric results
    results = {
        "S0_est": S0_est,
        "mu_hat": mu_hat,
        "sigma_hat": sigma_hat,
        "mc_call_price": price_call_mc,
        "mc_call_se": se_call_mc,
        "mc_put_price": price_put_mc,
        "mc_put_se": se_put_mc,
        "bs_call_price": bs_call,
        "bs_put_price": bs_put,
        "lsm_american_put": lsm_price,
        "delta_mc_fd": delta_mc,
        "gamma_mc_fd": gamma_mc,
        "vega_mc_fd": vega_mc,
        "rho_mc_fd": rho_mc,
        "theta_mc_fd": theta_mc,
        "delta_pathwise": delta_pw
    }
    df_results = pd.DataFrame([results])
    out_csv = os.path.join(out_dir, "results_summary.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"Saved summary results to {out_csv}")

    print("\nAll done, check the images and CSV in the quant_outputs folder. Modify parameters at the top and re-run to experiment.")

if __name__ == "__main__":
    main()
