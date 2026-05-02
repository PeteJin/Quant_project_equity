import os
import sys
import contextlib
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


project1_folder = os.path.dirname(__file__)
if project1_folder not in sys.path:
    sys.path.append(project1_folder)

old_show = plt.show
plt.show = lambda *args, **kwargs: None

with contextlib.redirect_stdout(io.StringIO()):
    try:
        from equity_quant_project.project1 import bl
        from equity_quant_project.project1 import optimizers
    except ModuleNotFoundError:
        import bl
        import optimizers

plt.show = old_show


assets = bl.assets
output_folder = bl.output_folder

n_paths = 500
horizon_days = 252
start_value = 1.0

mu_daily = bl.bl_return.reindex(assets) / 252
covariance_daily = bl.bl_covariance.reindex(index=assets, columns=assets) / 252
cholesky_matrix = np.linalg.cholesky(covariance_daily.values)

weight_table = pd.DataFrame({
    "BL": bl.all_weights.reindex(assets).fillna(0),
    "MVO Min Vol": optimizers.mvo_minvol.reindex(assets).fillna(0),
    "MVO Semivariance": optimizers.mvo_semivariance.reindex(assets).fillna(0),
    "MVO CVaR": optimizers.mvo_cvar.reindex(assets).fillna(0),
    "HRP": optimizers.hrp_weights.reindex(assets).fillna(0),
    "NCO": optimizers.nco_weights.reindex(assets).fillna(0),
})

weight_table = weight_table.div(weight_table.sum(axis=0), axis=1)

np.random.seed(2026)

simulation_results = {}
summary = pd.DataFrame(
    index=weight_table.columns,
    columns=[
        "mean_terminal_value",
        "mean_1y_return",
        "vol_terminal_value",
        "var_5_loss",
        "probability_of_loss",
        "average_max_drawdown_loss",
        "median_max_drawdown_loss",
    ],
)

for allocation_name in weight_table.columns:
    weights = weight_table[allocation_name].values
    portfolio_paths = np.zeros((horizon_days + 1, n_paths))
    portfolio_paths[0, :] = start_value

    for day in range(1, horizon_days + 1):
        random_normals = np.random.normal(size=(n_paths, len(assets)))
        correlated_randoms = random_normals @ cholesky_matrix.T
        log_returns = mu_daily.values - 0.5 * np.diag(covariance_daily.values) + correlated_randoms
        asset_returns = np.exp(log_returns) - 1
        portfolio_daily_returns = asset_returns @ weights
        portfolio_paths[day, :] = portfolio_paths[day - 1, :] * (1 + portfolio_daily_returns)

    simulation_results[allocation_name] = portfolio_paths

    terminal_values = portfolio_paths[-1, :]
    terminal_returns = terminal_values / start_value - 1
    running_max = np.maximum.accumulate(portfolio_paths, axis=0)
    drawdowns = portfolio_paths / running_max - 1
    max_drawdown = drawdowns.min(axis=0)
    terminal_var_5 = np.quantile(terminal_returns, 0.05)
    max_drawdown_loss = -max_drawdown

    summary.loc[allocation_name, "mean_terminal_value"] = terminal_values.mean()
    summary.loc[allocation_name, "mean_1y_return"] = terminal_returns.mean()
    summary.loc[allocation_name, "vol_terminal_value"] = terminal_values.std()
    summary.loc[allocation_name, "var_5_loss"] = max(0, -terminal_var_5)
    summary.loc[allocation_name, "probability_of_loss"] = (terminal_returns < 0).mean()
    summary.loc[allocation_name, "average_max_drawdown_loss"] = max_drawdown_loss.mean()
    summary.loc[allocation_name, "median_max_drawdown_loss"] = np.median(max_drawdown_loss)

summary = summary.astype(float)

os.makedirs(output_folder, exist_ok=True)
summary.to_csv(output_folder + "/monte_carlo_summary.csv")
weight_table.to_csv(output_folder + "/monte_carlo_weights.csv")

print("Cholesky matrix:")
print(pd.DataFrame(cholesky_matrix, index=assets, columns=assets).round(5).to_string())
print()
print("Monte Carlo summary:")
print(summary.round(4).to_string())


selected_allocations = ["BL", "MVO CVaR", "HRP", "NCO"]

for allocation_name in weight_table.columns:
    portfolio_paths = simulation_results[allocation_name]
    fig, ax = plt.subplots(figsize=(9.5, 5.4))

    path_5 = np.percentile(portfolio_paths, 5, axis=1)
    path_50 = np.percentile(portfolio_paths, 50, axis=1)
    path_95 = np.percentile(portfolio_paths, 95, axis=1)

    ax.plot(portfolio_paths, color="#1f77b4", alpha=0.16, linewidth=1.15)
    ax.fill_between(np.arange(horizon_days + 1), path_5, path_95, color="#1f77b4", alpha=0.12, linewidth=0)
    ax.plot(path_50, color="black", linewidth=2.6, label="Median path")
    ax.plot(portfolio_paths.mean(axis=1), color="#d62728", linewidth=2.0, linestyle="--", label="Average path")
    ax.axhline(1.0, color="gray", linewidth=1.0)

    ax.set_title(allocation_name + " GBM Random Walks", fontsize=17, fontweight="bold", pad=12)
    ax.set_xlabel("Trading days")
    ax.set_ylabel("Portfolio value")
    ax.legend(frameon=False)
    ax.grid(False)
    y_min = portfolio_paths.min()
    y_max = portfolio_paths.max()
    y_margin = (y_max - y_min) * 0.06
    ax.set_ylim(max(0, y_min - y_margin), y_max + y_margin)

    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_linewidth(1.2)
        ax.spines[side].set_color("black")

    fig.tight_layout()
    plt.show(block=False)


fig2, axes = plt.subplots(2, 3, figsize=(13, 7))
axes = axes.ravel()

for i, allocation_name in enumerate(weight_table.columns):
    portfolio_paths = simulation_results[allocation_name]
    terminal_returns = portfolio_paths[-1, :] - 1
    ax2 = axes[i]
    ax2.hist(terminal_returns, bins=22, color="#4c78a8", alpha=0.78, edgecolor="white")
    ax2.axvline(terminal_returns.mean(), color="black", linewidth=1.6, label="Mean")
    ax2.axvline(np.quantile(terminal_returns, 0.05), color="red", linewidth=1.4, linestyle="--", label="5%")
    ax2.axvline(0, color="gray", linewidth=1.0)
    ax2.set_title(allocation_name, fontsize=13, fontweight="bold")
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.grid(False)

    for side in ["top", "right", "bottom", "left"]:
        ax2.spines[side].set_linewidth(1.1)
        ax2.spines[side].set_color("black")

fig2.suptitle("Monte Carlo Terminal Return Distributions", fontsize=18, fontweight="bold")

fig2.tight_layout()
plt.show(block=False)


fig3, ax3 = plt.subplots(figsize=(10, 5.5))
summary[["var_5_loss", "average_max_drawdown_loss", "median_max_drawdown_loss"]].plot(kind="bar", ax=ax3, width=0.72)

ax3.set_title("Monte Carlo Downside Risk Summary", fontsize=18, fontweight="bold", pad=12)
ax3.set_ylabel("Loss magnitude")
ax3.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax3.legend(["5% VaR", "Average max drawdown", "Median max drawdown"], frameon=False, fontsize=9)
ax3.grid(False)
ax3.tick_params(axis="x", labelrotation=30, labelsize=10, width=1.1)
ax3.tick_params(axis="y", labelsize=10, width=1.1)

for side in ["top", "right", "bottom", "left"]:
    ax3.spines[side].set_linewidth(1.2)
    ax3.spines[side].set_color("black")

fig3.tight_layout()
plt.show(block=False)
