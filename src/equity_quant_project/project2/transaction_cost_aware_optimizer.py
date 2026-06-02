import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


project1_folder = "data/processed/project1"
output_folder = "data/processed/project2"
figure_folder = "reports/figures/project2"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(figure_folder, exist_ok=True)

portfolio_value = 10000000
start_date = "2010-01-01"
volatility_span = 252
volume_window = 252
T = 30
halflife = 5
forecast = 50
kappa = 1e-7
execution_kappa = 3e-4
Theta = 2e8
frontier_kappa_values = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]

cost_tiers = pd.DataFrame(
    {
        "tier": ["Low", "Medium", "High"],
        "gamma": [0.20, 0.314, 0.45],
        "eta": [0.08, 0.142, 0.22],
        "beta": [0.55, 0.60, 0.65],
    }
).set_index("tier")

tier_by_asset = {
    "TLT": "Low",
    "SHY": "Low",
    "GLD": "Low",
    "MS": "Low",
    "XOM": "Low",
    "LMT": "Medium",
    "DE": "Medium",
    "MO": "Medium",
    "ORLY": "Medium",
    "CASY": "High",
    "EME": "High",
}

tier_volume = {
    "Low": 8000000,
    "Medium": 2500000,
    "High": 500000,
}


def alpha(t, forecast, halflife):
    return 1e-4 * forecast * 2 ** (-t / halflife)


def cost(delta, P, gamma, sigma, V, Theta, eta, beta):
    X = delta / P
    return P * X * (
        gamma * sigma / 2 * X / V * (Theta / V) ** 0.25
        + np.sign(X) * eta * sigma * np.abs(X / V) ** beta
    )


def minusutility(x1, x0, t, order_shares, side, kappa, forecast, halflife, P, gamma, sigma, V, Theta, eta, beta):
    delta = side * (x1 - x0) * P
    remaining = order_shares - x1
    opportunity = remaining * P * alpha(t, forecast, halflife)
    risks = kappa / 2 * sigma ** 2 * remaining ** 2
    costs = abs(cost(delta, P, gamma, sigma, V, Theta, eta, beta))
    return opportunity + risks + costs


def optimizer(order_shares, side, T, kappa, forecast, halflife, P, gamma, sigma, V, Theta, eta, beta):
    tau = 1 / T
    time_grid = np.arange(T + 1)
    price_sigma = P * sigma
    impact_scale = eta * sigma * P / max(V ** beta, 1)
    speed = np.sqrt(kappa * price_sigma ** 2 / max(impact_scale, 0.000001))

    if speed < 0.000001:
        remaining = order_shares * (T - time_grid) / T
    else:
        remaining = order_shares * np.sinh(speed * tau * (T - time_grid)) / np.sinh(speed * tau * T)

    path = order_shares - remaining
    path[0] = 0
    path[-1] = order_shares
    return path


def execution_profit(path, side, forecast, halflife, P, gamma, sigma, V, Theta, eta, beta):
    times = np.arange(path.shape[0])
    delta = np.diff(path) * side * P
    alpha_profit = np.array([x * P * alpha(t, forecast, halflife) for x, t in zip(path, times)])
    execution_costs = np.array([abs(cost(d, P, gamma, sigma, V, Theta, eta, beta)) for d in delta])
    return np.sum(alpha_profit) - np.sum(execution_costs)


def execution_sharpe(path, side, forecast, halflife, P, gamma, sigma, V, Theta, eta, beta):
    expected_profit = execution_profit(path, side, forecast, halflife, P, gamma, sigma, V, Theta, eta, beta)
    risk = np.linalg.norm(path) * P * sigma
    return np.sqrt(252) * expected_profit / max(risk, 0.000001)


def execution_risk(path, P, sigma):
    remaining = path[-1] - path
    return P * sigma * np.sqrt(np.sum(remaining[:-1] ** 2) / T)


weights = pd.read_csv(project1_folder + "/rolling_bl_weight_history.csv", index_col=0)
weights.index = pd.to_datetime(weights.index)

weight_change = weights.diff().abs().sum(axis=1)
rebalance_date = weight_change.idxmax()
previous_date = weights.index[weights.index.get_loc(rebalance_date) - 1]

current_weight = weights.loc[previous_date]
target_weight = weights.loc[rebalance_date]
trade_weight = target_weight - current_weight

assets = list(weights.columns)

prices = pd.read_csv(project1_folder + "/project1_prices.csv", index_col=0)
prices.index = pd.to_datetime(prices.index)
prices = prices.reindex(columns=assets).dropna()
returns = prices.pct_change().dropna()

price_at_rebalance = prices.loc[:rebalance_date].iloc[-1]
latest_prices = prices.iloc[-1]
sigma = returns.ewm(span=volatility_span).std().iloc[-1]
liquidity_tier = pd.Series(tier_by_asset).reindex(assets)
average_volume = liquidity_tier.map(tier_volume)
dollar_volume = latest_prices * average_volume

trade_list = pd.DataFrame(index=assets)
trade_list["current_weight"] = current_weight
trade_list["target_weight"] = target_weight
trade_list["trade_weight"] = trade_weight
trade_list["trade_dollar"] = trade_list["trade_weight"] * portfolio_value
trade_list["price"] = price_at_rebalance
trade_list["shares"] = trade_list["trade_dollar"] / trade_list["price"]
trade_list["abs_shares"] = trade_list["shares"].abs()
trade_list["side"] = np.where(trade_list["shares"] >= 0, 1, -1)
trade_list["ewma_daily_volatility"] = sigma
trade_list["average_daily_volume"] = average_volume
trade_list["dollar_volume"] = dollar_volume
trade_list["liquidity_tier"] = liquidity_tier
trade_list = trade_list[trade_list["abs_shares"] > 1]

paths = pd.DataFrame(index=range(T + 1))
path_progress = pd.DataFrame(index=range(T + 1))
trades = pd.DataFrame(index=range(1, T + 1))
summary = pd.DataFrame(
    index=trade_list.index,
    columns=[
        "liquidity_tier",
        "trade_dollar",
        "shares",
        "expected_profit",
        "execution_cost",
        "execution_cost_bps",
        "execution_sharpe",
    ],
)

for ticker in trade_list.index:
    tier = trade_list.loc[ticker, "liquidity_tier"]
    P = trade_list.loc[ticker, "price"]
    asset_sigma = trade_list.loc[ticker, "ewma_daily_volatility"]
    V = trade_list.loc[ticker, "average_daily_volume"]
    gamma = cost_tiers.loc[tier, "gamma"]
    eta = cost_tiers.loc[tier, "eta"]
    beta = cost_tiers.loc[tier, "beta"]
    order_shares = trade_list.loc[ticker, "abs_shares"]
    side = trade_list.loc[ticker, "side"]

    path = optimizer(order_shares, side, T, execution_kappa, forecast, halflife, P, gamma, asset_sigma, V, Theta, eta, beta)
    child_trades = np.diff(path) * side
    path_signed = path * side

    execution_cost = np.sum([abs(cost(d * P, P, gamma, asset_sigma, V, Theta, eta, beta)) for d in child_trades])
    expected_profit = execution_profit(path, side, forecast, halflife, P, gamma, asset_sigma, V, Theta, eta, beta)
    sharpe = execution_sharpe(path, side, forecast, halflife, P, gamma, asset_sigma, V, Theta, eta, beta)

    paths[ticker] = path_signed
    path_progress[ticker] = path / order_shares
    trades[ticker] = child_trades
    summary.loc[ticker, "liquidity_tier"] = tier
    summary.loc[ticker, "trade_dollar"] = trade_list.loc[ticker, "trade_dollar"]
    summary.loc[ticker, "shares"] = trade_list.loc[ticker, "shares"]
    summary.loc[ticker, "expected_profit"] = expected_profit
    summary.loc[ticker, "execution_cost"] = execution_cost
    summary.loc[ticker, "execution_cost_bps"] = execution_cost / abs(trade_list.loc[ticker, "trade_dollar"]) * 10000
    summary.loc[ticker, "execution_sharpe"] = sharpe


summary[["trade_dollar", "shares", "expected_profit", "execution_cost", "execution_cost_bps", "execution_sharpe"]] = summary[
    ["trade_dollar", "shares", "expected_profit", "execution_cost", "execution_cost_bps", "execution_sharpe"]
].astype(float)

paths.index.name = "trade_step"
path_progress.index.name = "trade_step"
trades.index.name = "trade_step"
trade_list.index.name = "ticker"
summary.index.name = "ticker"

trade_list.to_csv(output_folder + "/rebalance_trade_list.csv")
paths.to_csv(output_folder + "/optimal_execution_paths.csv")
path_progress.to_csv(output_folder + "/optimal_execution_progress.csv")
trades.to_csv(output_folder + "/optimal_child_trades.csv")
summary.to_csv(output_folder + "/execution_cost_summary.csv")

print("Selected rebalance date:", rebalance_date.date())
print("Previous rebalance date:", previous_date.date())
print("Total traded notional:", "${:,.0f}".format(trade_list["trade_dollar"].abs().sum()))
print()
print("Execution summary:")
print(summary.sort_values("execution_cost_bps", ascending=False).round(4).to_string())


fig, ax = plt.subplots(figsize=(11, 6.5))

for ticker in path_progress.columns:
    ax.plot(path_progress.index, path_progress[ticker], linewidth=2.0, label=ticker)

ax.set_title("Optimal Execution Paths by Asset", fontsize=18, fontweight="bold", pad=12)
ax.set_xlabel("Trade step", fontsize=12)
ax.set_ylabel("Execution progress", fontsize=12)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend(frameon=False, ncol=2, fontsize=9)
ax.grid(False)

for side in ["top", "right", "bottom", "left"]:
    ax.spines[side].set_linewidth(1.2)
    ax.spines[side].set_color("black")

fig.tight_layout()
fig.savefig(figure_folder + "/Optimal Execution Paths by Asset.png", dpi=160)
fig.savefig(figure_folder + "/Almgren-Chriss Execution Paths.png", dpi=160)
plt.show(block=False)


fig2, ax2 = plt.subplots(figsize=(10, 6.5))

plot_summary = summary.sort_values("execution_cost_bps", ascending=False)
colors = plot_summary["liquidity_tier"].map({"Low": "#2ca02c", "Medium": "#ff7f0e", "High": "#d62728"})
ax2.bar(plot_summary.index, plot_summary["execution_cost_bps"], color=colors)

ax2.set_title("Execution Cost by Asset", fontsize=18, fontweight="bold", pad=12)
ax2.set_xlabel("Ticker", fontsize=12)
ax2.set_ylabel("Cost in bps of traded notional", fontsize=12)
ax2.tick_params(axis="x", labelrotation=35)
ax2.grid(False)

for side in ["top", "right", "bottom", "left"]:
    ax2.spines[side].set_linewidth(1.2)
    ax2.spines[side].set_color("black")

fig2.tight_layout()
fig2.savefig(figure_folder + "/Execution Cost by Asset.png", dpi=160)
plt.show(block=False)


kappa_values = np.array(frontier_kappa_values)
frontier = pd.DataFrame(index=kappa_values, columns=["expected_profit", "execution_cost", "risk"])

for frontier_kappa in kappa_values:
    total_profit = 0.0
    total_cost = 0.0
    total_risk = 0.0

    for ticker in trade_list.index:
        tier = trade_list.loc[ticker, "liquidity_tier"]
        P = trade_list.loc[ticker, "price"]
        asset_sigma = trade_list.loc[ticker, "ewma_daily_volatility"]
        V = trade_list.loc[ticker, "average_daily_volume"]
        gamma = cost_tiers.loc[tier, "gamma"]
        eta = cost_tiers.loc[tier, "eta"]
        beta = cost_tiers.loc[tier, "beta"]
        order_shares = trade_list.loc[ticker, "abs_shares"]
        side = trade_list.loc[ticker, "side"]
        path = optimizer(order_shares, side, T, frontier_kappa, forecast, halflife, P, gamma, asset_sigma, V, Theta, eta, beta)
        child_trades = np.diff(path) * side

        total_profit += execution_profit(path, side, forecast, halflife, P, gamma, asset_sigma, V, Theta, eta, beta)
        total_cost += np.sum([abs(cost(d * P, P, gamma, asset_sigma, V, Theta, eta, beta)) for d in child_trades])
        total_risk += execution_risk(path, P, asset_sigma)

    frontier.loc[frontier_kappa, "expected_profit"] = total_profit
    frontier.loc[frontier_kappa, "execution_cost"] = total_cost
    frontier.loc[frontier_kappa, "risk"] = total_risk

frontier = frontier.astype(float)
frontier.to_csv(output_folder + "/execution_cost_risk_frontier.csv")

fig3, ax3 = plt.subplots(figsize=(9.5, 6.5))
ax3.plot(frontier["risk"], frontier["execution_cost"], marker="o", linewidth=2.0)

ax3.set_title("Execution Cost-Risk Frontier", fontsize=18, fontweight="bold", pad=12)
ax3.set_xlabel("Execution risk", fontsize=12)
ax3.set_ylabel("Execution cost", fontsize=12)
ax3.margins(x=0.08, y=0.18)
ax3.grid(False)

for side in ["top", "right", "bottom", "left"]:
    ax3.spines[side].set_linewidth(1.2)
    ax3.spines[side].set_color("black")

fig3.tight_layout()
fig3.savefig(figure_folder + "/Execution Cost-Risk Frontier.png", dpi=160)
fig3.savefig(figure_folder + "/Execution Cost-Risk Tradeoff.png", dpi=160)
plt.show(block=False)
