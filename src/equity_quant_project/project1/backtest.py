import os

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from hmmlearn.hmm import GaussianHMM
from pypfopt import EfficientFrontier
from pypfopt import black_litterman
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.hierarchical_portfolio import HRPOpt
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform


assets = ["CASY", "ORLY", "TLT", "SHY", "LMT", "DE", "MO", "GLD", "MS", "XOM", "EME"]

raw_folder = "data/raw/macro"
output_folder = "data/processed/project1"

macro_start = "1997-01-01"
price_start = "2005-01-01"
backtest_start = "2010-01-01"
risk_free_rate = 0.04
max_weight = 0.40
regime_threshold = 0.70
view_uncertainty = 0.20
smoothing_window = 126
posterior_sharpness = 1.0
variance_scale = 1.4
active_sleeve_weight = 0.90
active_score_power = 4.0
top_active_names = 4


vix = pd.read_csv(raw_folder + "/VIX.csv", encoding="latin1")
vix.columns = vix.columns.str.replace(chr(239) + chr(187) + chr(191), "").str.replace('"', "").str.strip()
vix = vix[["DATE", "CLOSE"]]
vix["DATE"] = pd.to_datetime(vix["DATE"], errors="coerce")
vix["CLOSE"] = pd.to_numeric(vix["CLOSE"], errors="coerce")
vix = vix.dropna()
vix = vix.set_index("DATE").sort_index()
vix = vix.pct_change()
vix.columns = ["vix"]

spread = pd.read_csv(raw_folder + "/High Yield Spread.csv", encoding="latin1")
spread.columns = spread.columns.str.replace(chr(239) + chr(187) + chr(191), "").str.replace('"', "").str.strip()
spread = spread[["Date", "Value"]]
spread["Date"] = pd.to_datetime(spread["Date"], errors="coerce")
spread["Value"] = pd.to_numeric(spread["Value"], errors="coerce")
spread = spread.dropna()
spread = spread.set_index("Date").sort_index()
spread = spread.diff()
spread.columns = ["high_yield_spread"]

curve = pd.read_csv(raw_folder + "/T10Y2Y.csv", encoding="latin1")
curve.columns = curve.columns.str.replace(chr(239) + chr(187) + chr(191), "").str.replace('"', "").str.strip()
curve = curve[["observation_date", "T10Y2Y"]]
curve["observation_date"] = pd.to_datetime(curve["observation_date"], errors="coerce")
curve["T10Y2Y"] = pd.to_numeric(curve["T10Y2Y"], errors="coerce")
curve = curve.dropna()
curve = curve.set_index("observation_date").sort_index()
curve = curve.diff()
curve.columns = ["t10y2y"]

macro_data = pd.concat([vix, spread, curve], axis=1)
macro_data = macro_data[macro_data.index >= macro_start]
macro_data = macro_data.replace([np.inf, -np.inf], np.nan)
macro_data = macro_data.ffill().dropna()


download_tickers = assets + ["SPY"]

all_prices = yf.download(
    download_tickers,
    start=price_start,
    auto_adjust=True,
    threads=False,
    progress=False,
)["Close"]

if isinstance(all_prices, pd.Series):
    all_prices = all_prices.to_frame()

prices = all_prices.reindex(columns=assets)
prices = prices.dropna()
daily_returns = prices.pct_change().dropna()

spy_prices = all_prices["SPY"].dropna()

market_caps = pd.Series(1.0, index=assets)

month_end_prices = prices.resample("ME").last()
rebalance_dates = month_end_prices.index[month_end_prices.index >= pd.Timestamp(backtest_start)]

strategy_names = ["BL", "MVO Min Vol", "HRP", "NCO", "Equal Weight", "S&P 500"]
realized_returns = pd.DataFrame(index=rebalance_dates[1:], columns=strategy_names)
turnover = pd.Series(0.0, index=rebalance_dates[1:])
weight_history = {}
previous_bl_weight = pd.Series(1 / len(assets), index=assets)
daily_plot_returns = {name: [] for name in strategy_names}

for i in range(len(rebalance_dates) - 1):
    train_end = rebalance_dates[i]
    hold_end = rebalance_dates[i + 1]

    macro_train = macro_data[macro_data.index <= train_end].copy()
    macro_z = (macro_train - macro_train.mean()) / macro_train.std(ddof=0)
    macro_z = macro_z.dropna()

    tightness_raw = macro_z["vix"] + macro_z["high_yield_spread"] - macro_z["t10y2y"]
    tightness_index = tightness_raw.rolling(smoothing_window).mean()
    tightness_index = tightness_index.dropna()
    macro_tightness = (tightness_index - tightness_index.mean()) / tightness_index.std(ddof=0)

    X = tightness_raw.loc[macro_tightness.index].values.reshape(-1, 1)
    state_names = ["Loose", "Tight"]
    starting_state = (macro_tightness >= 0).astype(int)
    start_probability = np.array([
        (starting_state == 0).mean(),
        (starting_state == 1).mean(),
    ])

    transition_matrix_start = pd.DataFrame(0.5, index=state_names, columns=state_names)

    for j in range(len(starting_state) - 1):
        old_state = state_names[starting_state.iloc[j]]
        new_state = state_names[starting_state.iloc[j + 1]]
        transition_matrix_start.loc[old_state, new_state] = transition_matrix_start.loc[old_state, new_state] + 1

    persistence_penalty = 9000
    transition_matrix_start.loc["Loose", "Loose"] += persistence_penalty
    transition_matrix_start.loc["Tight", "Tight"] += persistence_penalty
    transition_matrix_start = transition_matrix_start.div(transition_matrix_start.sum(axis=1), axis=0)

    mean_start = np.array([
        X[starting_state == 0].mean(),
        X[starting_state == 1].mean(),
    ]).reshape(2, 1)

    variance_start = np.array([
        X[starting_state == 0].var(),
        X[starting_state == 1].var(),
    ]).reshape(2, 1)

    model = GaussianHMM(
        n_components=2,
        covariance_type="diag",
        n_iter=300,
        tol=0.001,
        init_params="",
        params="smc",
    )
    model.startprob_ = start_probability
    model.transmat_ = transition_matrix_start.values
    model.means_ = mean_start
    model.covars_ = variance_start
    model.fit(X)

    posterior = model.predict_proba(X)
    state_order = np.argsort(model.means_.ravel())
    posterior = posterior[:, state_order]
    posterior = posterior ** posterior_sharpness
    posterior = posterior / posterior.sum(axis=1, keepdims=True)

    regime_probability = pd.DataFrame(posterior, index=macro_tightness.index, columns=state_names)
    transition_matrix = model.transmat_[state_order][:, state_order]

    current_regime_probability = regime_probability.iloc[-1].values
    forward_regime_probability = np.zeros(2)

    for j in range(1, 253):
        forward_regime_probability = forward_regime_probability + current_regime_probability @ np.linalg.matrix_power(transition_matrix, j)

    forward_regime_probability = forward_regime_probability / 252
    loose_forward_probability = forward_regime_probability[0]
    tight_forward_probability = forward_regime_probability[1]

    stock_train = daily_returns[daily_returns.index <= train_end].copy()
    forward_stock_returns = stock_train.shift(-1)
    common_dates = forward_stock_returns.index.intersection(regime_probability.index)
    regime_stock_returns = forward_stock_returns.loc[common_dates]
    regime_probability_stock = regime_probability.loc[common_dates]
    regime_stock_returns = regime_stock_returns.iloc[:-1].dropna()
    regime_probability_stock = regime_probability_stock.loc[regime_stock_returns.index]

    loose_days = regime_probability_stock["Loose"] >= regime_threshold
    tight_days = regime_probability_stock["Tight"] >= regime_threshold

    loose_returns = regime_stock_returns.loc[loose_days].dropna()
    tight_returns = regime_stock_returns.loc[tight_days].dropna()
    base_returns = stock_train.dropna()

    if len(loose_returns) < 126:
        loose_returns = base_returns

    if len(tight_returns) < 126:
        tight_returns = base_returns

    loose_mu = loose_returns.mean() * 252
    tight_mu = tight_returns.mean() * 252
    loose_cov = loose_returns.cov() * 252
    tight_cov = tight_returns.cov() * 252

    mu = loose_mu * loose_forward_probability + tight_mu * tight_forward_probability
    S = loose_cov * loose_forward_probability + tight_cov * tight_forward_probability
    S = S.reindex(index=assets, columns=assets)
    mu = mu.reindex(assets)

    diagonal_fix = pd.Series(np.diag(S.values), index=assets)
    diagonal_fix[diagonal_fix <= 0] = stock_train.var().reindex(assets) * 252

    for ticker in assets:
        S.loc[ticker, ticker] = diagonal_fix[ticker]

    spy_train = spy_prices[spy_prices.index <= train_end]
    delta = black_litterman.market_implied_risk_aversion(spy_train, risk_free_rate=risk_free_rate)
    omega = pd.DataFrame(np.diag((np.sqrt(np.diag(S.values)) * view_uncertainty) ** 2), index=assets, columns=assets)
    prior_return = black_litterman.market_implied_prior_returns(market_caps, delta, S, risk_free_rate=risk_free_rate)

    bl_model = BlackLittermanModel(
        S,
        pi=prior_return,
        absolute_views=mu.to_dict(),
        omega=omega,
        tau=0.05,
    )

    bl_return = bl_model.bl_returns()
    bl_covariance = bl_model.bl_cov()

    try:
        ef = EfficientFrontier(bl_return, bl_covariance, weight_bounds=(0, max_weight))
        ef.max_sharpe(risk_free_rate=risk_free_rate)
        bl_weight = pd.Series(ef.clean_weights()).reindex(assets).fillna(0)
    except Exception:
        ef = EfficientFrontier(bl_return, bl_covariance, weight_bounds=(0, max_weight))
        ef.min_volatility()
        bl_weight = pd.Series(ef.clean_weights()).reindex(assets).fillna(0)

    bl_weight[bl_weight < 0.0001] = 0
    bl_weight = bl_weight / bl_weight.sum()

    ef = EfficientFrontier(bl_return, bl_covariance, weight_bounds=(0, max_weight))
    try:
        ef.min_volatility()
    except Exception:
        ef = EfficientFrontier(bl_return, bl_covariance, weight_bounds=(0, max_weight))
        ef.min_volatility()
    mvo_weight = pd.Series(ef.clean_weights()).reindex(assets).fillna(0)
    mvo_weight[mvo_weight < 0.0001] = 0
    mvo_weight = mvo_weight / mvo_weight.sum()

    try:
        combined_regime_returns = pd.concat([loose_returns, tight_returns]).sort_index()
        combined_regime_returns = combined_regime_returns[~combined_regime_returns.index.duplicated(keep="first")]
        hrp = HRPOpt(combined_regime_returns)
        hrp.optimize()
        hrp_weight = pd.Series(hrp.clean_weights()).reindex(assets).fillna(0)
    except Exception:
        hrp_weight = mvo_weight.copy()

    hrp_weight[hrp_weight < 0.0001] = 0
    hrp_weight = hrp_weight / hrp_weight.sum()

    correlation = combined_regime_returns.corr().reindex(index=assets, columns=assets)
    distance_matrix = np.sqrt(0.5 * (1 - correlation))
    distance_matrix = distance_matrix.replace([np.inf, -np.inf], np.nan).fillna(0)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix.values, 0)
    linkage_matrix = linkage(squareform(distance_matrix), method="ward")
    cluster_labels = fcluster(linkage_matrix, 3, criterion="maxclust")

    cluster_dict = {}

    for j, ticker in enumerate(assets):
        cluster_id = int(cluster_labels[j])

        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []

        cluster_dict[cluster_id].append(ticker)

    intra_weights = pd.DataFrame(0.0, index=assets, columns=cluster_dict.keys())

    for cluster_id, cluster_assets in cluster_dict.items():
        cluster_cov = bl_covariance.loc[cluster_assets, cluster_assets].values
        n_assets = len(cluster_assets)

        if n_assets == 1:
            cluster_weight = np.array([1.0])
        else:
            start_weights = np.ones(n_assets) / n_assets
            cluster_max_weight = min(1.0, max_weight * n_assets)
            bounds = tuple((0.0, cluster_max_weight) for _ in range(n_assets))
            constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}

            result = minimize(
                lambda x: x.T @ cluster_cov @ x,
                start_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )

            cluster_weight = result.x

        for j, ticker in enumerate(cluster_assets):
            intra_weights.loc[ticker, cluster_id] = cluster_weight[j]

    cluster_covariance = intra_weights.T @ bl_covariance @ intra_weights
    n_clusters = len(cluster_covariance)
    start_weights = np.ones(n_clusters) / n_clusters
    bounds = tuple((0.0, 1.0) for _ in range(n_clusters))
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}

    inter_result = minimize(
        lambda x: np.sum(
            (
                x * (cluster_covariance.values @ x) / (x.T @ cluster_covariance.values @ x)
                - np.ones(n_clusters) / n_clusters
            ) ** 2
        ),
        start_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    nco_weight = intra_weights.dot(inter_result.x)
    nco_weight[nco_weight < 0.0001] = 0
    nco_weight = nco_weight / nco_weight.sum()

    equal_weight = pd.Series(1 / len(assets), index=assets)

    hold_returns = daily_returns[(daily_returns.index > train_end) & (daily_returns.index <= hold_end)]
    weights = {
        "BL": bl_weight,
        "MVO Min Vol": mvo_weight,
        "HRP": hrp_weight,
        "NCO": nco_weight,
        "Equal Weight": equal_weight,
    }

    forecast_vol = pd.Series(np.sqrt(np.diag(bl_covariance.values)), index=assets)
    return_to_risk_score = (bl_return.reindex(assets) - risk_free_rate) / forecast_vol

    trailing_prices = prices[prices.index <= train_end].tail(253)
    trailing_momentum = trailing_prices.iloc[-1] / trailing_prices.iloc[0] - 1
    trailing_volatility = stock_train.tail(252).std() * np.sqrt(252)

    return_to_risk_score = return_to_risk_score.replace([np.inf, -np.inf], np.nan)
    trailing_momentum = trailing_momentum.replace([np.inf, -np.inf], np.nan)
    trailing_volatility = trailing_volatility.replace([np.inf, -np.inf], np.nan)

    active_score = (
        return_to_risk_score.rank(pct=True)
        + trailing_momentum.rank(pct=True)
        - trailing_volatility.rank(pct=True) * 0.25
    )
    active_score = active_score.reindex(assets).fillna(active_score.median())
    active_score = active_score - active_score.min() + 0.0001
    active_score = active_score ** active_score_power

    top_names = active_score.sort_values(ascending=False).head(top_active_names).index
    active_weight = pd.Series(0.0, index=assets)
    active_weight.loc[top_names] = active_score.loc[top_names] / active_score.loc[top_names].sum()

    for j in range(10):
        capped_weight = active_weight.clip(upper=max_weight)
        weight_gap = 1 - capped_weight.sum()

        if abs(weight_gap) < 0.000001:
            break

        open_names = capped_weight[capped_weight < max_weight].index

        if len(open_names) == 0:
            break

        capped_weight.loc[open_names] = capped_weight.loc[open_names] + weight_gap * capped_weight.loc[open_names] / capped_weight.loc[open_names].sum()
        active_weight = capped_weight.copy()

    active_weight = active_weight / active_weight.sum()

    for name in ["BL", "MVO Min Vol", "HRP", "NCO"]:
        tilted_weight = weights[name] * (1 - active_sleeve_weight) + active_weight * active_sleeve_weight

        for j in range(10):
            capped_weight = tilted_weight.clip(upper=max_weight)
            weight_gap = 1 - capped_weight.sum()

            if abs(weight_gap) < 0.000001:
                break

            open_names = capped_weight[capped_weight < max_weight].index

            if len(open_names) == 0:
                break

            capped_weight.loc[open_names] = capped_weight.loc[open_names] + weight_gap * capped_weight.loc[open_names] / capped_weight.loc[open_names].sum()
            tilted_weight = capped_weight.copy()

        weights[name] = tilted_weight / tilted_weight.sum()

    for name in strategy_names:
        if name == "S&P 500":
            continue

        portfolio_daily = hold_returns @ weights[name]
        daily_plot_returns[name].append(portfolio_daily)
        realized_returns.loc[hold_end, name] = (1 + portfolio_daily).prod() - 1

    spy_hold_returns = spy_prices.pct_change().dropna()
    spy_hold_returns = spy_hold_returns[(spy_hold_returns.index > train_end) & (spy_hold_returns.index <= hold_end)]
    daily_plot_returns["S&P 500"].append(spy_hold_returns)
    spy_month_return = (1 + spy_hold_returns).prod() - 1
    realized_returns.loc[hold_end, "S&P 500"] = spy_month_return

    turnover.loc[hold_end] = abs(bl_weight - previous_bl_weight).sum() / 2
    previous_bl_weight = bl_weight.copy()
    weight_history[hold_end] = bl_weight


realized_returns = realized_returns.astype(float).dropna()
portfolio_values = (1 + realized_returns).cumprod()
cumulative_returns = portfolio_values - 1

daily_realized_returns = pd.DataFrame({
    name: pd.concat(daily_plot_returns[name]).sort_index()
    for name in strategy_names
})
daily_portfolio_values = (1 + daily_realized_returns).cumprod()
daily_cumulative_returns = daily_portfolio_values - 1

summary_columns = [
    "annual_return",
    "annual_volatility",
    "excess_return_vs_equal_weight",
    "excess_return_vs_sp500",
    "sharpe",
    "sortino",
    "max_drawdown",
    "time_to_recover_months",
    "monthly_var_5pct",
    "monthly_cvar_5pct",
]
summary = pd.DataFrame(index=strategy_names, columns=summary_columns)

for name in strategy_names:
    strategy_returns = realized_returns[name].dropna()
    portfolio_path = (1 + strategy_returns).cumprod()
    running_high = portfolio_path.cummax()
    drawdown = portfolio_path / running_high - 1
    var_5pct = strategy_returns.quantile(0.05)
    cvar_5pct = strategy_returns[strategy_returns <= var_5pct].mean()
    downside_returns = strategy_returns[strategy_returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(12)

    summary.loc[name, "annual_return"] = strategy_returns.mean() * 12
    summary.loc[name, "annual_volatility"] = strategy_returns.std() * np.sqrt(12)
    summary.loc[name, "sharpe"] = (summary.loc[name, "annual_return"] - risk_free_rate) / summary.loc[name, "annual_volatility"]
    summary.loc[name, "sortino"] = (summary.loc[name, "annual_return"] - risk_free_rate) / downside_volatility
    summary.loc[name, "max_drawdown"] = drawdown.min()
    summary.loc[name, "monthly_var_5pct"] = var_5pct
    summary.loc[name, "monthly_cvar_5pct"] = cvar_5pct

    drawdown_start = running_high.loc[:drawdown.idxmin()].idxmax()
    drawdown_low = drawdown.idxmin()
    recovery_path = portfolio_path[portfolio_path.index > drawdown_low]
    recovery_path = recovery_path[recovery_path >= running_high.loc[drawdown_start]]

    if len(recovery_path) > 0:
        recovery_date = recovery_path.index[0]
    else:
        recovery_date = portfolio_path.index[-1]

    summary.loc[name, "time_to_recover_months"] = len(portfolio_path[(portfolio_path.index > drawdown_low) & (portfolio_path.index <= recovery_date)])

summary = summary.astype(float)
summary["excess_return_vs_equal_weight"] = summary["annual_return"] - summary.loc["Equal Weight", "annual_return"]
summary["excess_return_vs_sp500"] = summary["annual_return"] - summary.loc["S&P 500", "annual_return"]
weight_history = pd.DataFrame(weight_history).T
weight_history.index.name = "date"

os.makedirs(output_folder, exist_ok=True)
realized_returns.to_csv(output_folder + "/rolling_backtest_returns.csv")
portfolio_values.to_csv(output_folder + "/rolling_backtest_values.csv")
cumulative_returns.to_csv(output_folder + "/rolling_backtest_cumulative_returns.csv")
daily_portfolio_values.to_csv(output_folder + "/rolling_backtest_daily_values.csv")
daily_cumulative_returns.to_csv(output_folder + "/rolling_backtest_daily_cumulative_returns.csv")
summary.to_csv(output_folder + "/rolling_backtest_summary.csv")
weight_history.to_csv(output_folder + "/rolling_bl_weight_history.csv")

return_table = summary[
    [
        "annual_return",
        "annual_volatility",
        "excess_return_vs_equal_weight",
        "excess_return_vs_sp500",
    ]
].copy()
return_table.columns = ["Ret", "Vol", "Ex EW", "Ex SPY"]

for column in return_table.columns:
    return_table[column] = return_table[column].map(lambda x: f"{x:.2%}")

risk_table = summary[
    [
        "sharpe",
        "sortino",
        "max_drawdown",
        "time_to_recover_months",
    ]
].copy()
risk_table.columns = ["Sharpe", "Sortino", "MDD", "Recovery"]
risk_table["Sharpe"] = risk_table["Sharpe"].map(lambda x: f"{x:.2f}")
risk_table["Sortino"] = risk_table["Sortino"].map(lambda x: f"{x:.2f}")
risk_table["MDD"] = risk_table["MDD"].map(lambda x: f"{x:.2%}")
risk_table["Recovery"] = risk_table["Recovery"].map(lambda x: f"{x:.0f}")

tail_table = summary[
    [
        "monthly_var_5pct",
        "monthly_cvar_5pct",
    ]
].copy()
tail_table.columns = ["VaR 5%", "CVaR 5%"]
tail_table["VaR 5%"] = tail_table["VaR 5%"].map(lambda x: f"{x:.2%}")
tail_table["CVaR 5%"] = tail_table["CVaR 5%"].map(lambda x: f"{x:.2%}")

print("Rolling backtest returns:")
print(return_table.to_string())
print()
print("Rolling backtest risk:")
print(risk_table.to_string())
print()
print("Rolling backtest tail risk:")
print(tail_table.to_string())
print()
print("Average BL turnover:", round(turnover.mean(), 4))
print()
print("Last BL allocation:")
print(weight_history.iloc[-1].sort_values(ascending=False).round(4).to_string())

fig, ax = plt.subplots(figsize=(12, 7))

for name in daily_cumulative_returns.columns:
    if name == "BL":
        ax.plot(daily_cumulative_returns.index, daily_cumulative_returns[name], label=name, linewidth=2.3)
    else:
        ax.plot(daily_cumulative_returns.index, daily_cumulative_returns[name], label=name, linewidth=1.9)

ax.set_title("Rolling Regime-Aware Portfolio Backtest", fontsize=20, fontweight="bold", pad=12)
ax.set_ylabel("Cumulative percentage change", fontsize=13)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend(frameon=False, fontsize=10)
ax.grid(False)
ax.tick_params(axis="x", labelrotation=0, labelsize=10, width=1.1)
ax.tick_params(axis="y", labelsize=10, width=1.1)

for side in ["top", "right", "bottom", "left"]:
    ax.spines[side].set_linewidth(1.2)
    ax.spines[side].set_color("black")

fig.tight_layout()
plt.show(block=False)

fig_compare, ax_compare = plt.subplots(figsize=(12, 7))

ax_compare.plot(daily_cumulative_returns.index, daily_cumulative_returns["BL"], label="BL", linewidth=2.4)
ax_compare.plot(daily_cumulative_returns.index, daily_cumulative_returns["S&P 500"], label="S&P 500", linewidth=2.0)
ax_compare.plot(daily_cumulative_returns.index, daily_cumulative_returns["Equal Weight"], label="Equal Weight", linewidth=2.0)

ax_compare.set_title("BL vs Benchmarks", fontsize=20, fontweight="bold", pad=12)
ax_compare.set_ylabel("Cumulative percentage change", fontsize=13)
ax_compare.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax_compare.legend(frameon=False, fontsize=10)
ax_compare.grid(False)
ax_compare.tick_params(axis="x", labelrotation=0, labelsize=10, width=1.1)
ax_compare.tick_params(axis="y", labelsize=10, width=1.1)

for side in ["top", "right", "bottom", "left"]:
    ax_compare.spines[side].set_linewidth(1.2)
    ax_compare.spines[side].set_color("black")

fig_compare.tight_layout()
plt.show(block=False)

fig2, ax2 = plt.subplots(figsize=(9, 6.5))
summary[["annual_return", "annual_volatility"]].plot(kind="bar", ax=ax2, width=0.72)

ax2.set_title("Rolling Backtest Return and Risk", fontsize=18, fontweight="bold", pad=12)
ax2.set_ylabel("Annualized value", fontsize=12)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax2.legend(["Return", "Volatility"], frameon=False, fontsize=10)
ax2.grid(False)
ax2.tick_params(axis="x", labelrotation=30, labelsize=10, width=1.1)
ax2.tick_params(axis="y", labelsize=10, width=1.1)

for side in ["top", "right", "bottom", "left"]:
    ax2.spines[side].set_linewidth(1.2)
    ax2.spines[side].set_color("black")

fig2.tight_layout()
plt.show(block=False)
