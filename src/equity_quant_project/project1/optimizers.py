import os
import sys
import contextlib
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pypfopt import EfficientCVaR
from pypfopt import EfficientFrontier
from pypfopt import EfficientSemivariance
from pypfopt import plotting
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform


project1_folder = os.path.dirname(__file__)
if project1_folder not in sys.path:
    sys.path.append(project1_folder)

old_show = plt.show
plt.show = lambda *args, **kwargs: None

with contextlib.redirect_stdout(io.StringIO()):
    try:
        from equity_quant_project.project1 import bl
    except ModuleNotFoundError:
        import bl

plt.show = old_show


assets = bl.assets
output_folder = bl.output_folder
risk_free_rate = bl.risk_free_rate
max_weight = bl.max_weight

bl_returns = bl.bl_return.reindex(assets)
covariance = bl.bl_covariance.reindex(index=assets, columns=assets)
bl_allocation = bl.all_weights.reindex(assets).fillna(0)
prices = bl.prices[assets].dropna()
returns = prices.pct_change().dropna()


# MVO
ef = EfficientFrontier(bl_returns, covariance, weight_bounds=(0, max_weight))
ef.min_volatility()
mvo_minvol = pd.Series(ef.clean_weights()).reindex(assets).fillna(0)
mvo_minvol[mvo_minvol < 0.0001] = 0
mvo_minvol = mvo_minvol / mvo_minvol.sum()

print("MVO minimum volatility performance:")
print("Expected annual return: {:.2f}%".format(100 * mvo_minvol.dot(bl_returns)))
print("Annual volatility: {:.2f}%".format(100 * np.sqrt(mvo_minvol.values @ covariance.values @ mvo_minvol.values)))
print("Sharpe ratio: {:.2f}".format((mvo_minvol.dot(bl_returns) - risk_free_rate) / np.sqrt(mvo_minvol.values @ covariance.values @ mvo_minvol.values)))
print("Return / volatility: {:.2f}".format(mvo_minvol.dot(bl_returns) / np.sqrt(mvo_minvol.values @ covariance.values @ mvo_minvol.values)))
print()

ef_semi = EfficientSemivariance(bl_returns, returns, weight_bounds=(0, max_weight))
ef_semi.efficient_return(0.20)
mvo_semivariance = pd.Series(ef_semi.clean_weights()).reindex(assets).fillna(0)
mvo_semivariance[mvo_semivariance < 0.0001] = 0
mvo_semivariance = mvo_semivariance / mvo_semivariance.sum()

print("MVO semivariance performance:")
ef_semi.portfolio_performance(verbose=True)
print()

ef_cvar = EfficientCVaR(bl_returns, returns, weight_bounds=(0, max_weight), beta=0.95)
ef_cvar.min_cvar()
mvo_cvar = pd.Series(ef_cvar.clean_weights()).reindex(assets).fillna(0)
mvo_cvar[mvo_cvar < 0.0001] = 0
mvo_cvar = mvo_cvar / mvo_cvar.sum()

print("MVO CVaR performance:")
print("Expected annual return: {:.2f}%".format(100 * mvo_cvar.dot(bl_returns)))
print("Annual volatility: {:.2f}%".format(100 * np.sqrt(mvo_cvar.values @ covariance.values @ mvo_cvar.values)))
print("Sharpe ratio: {:.2f}".format((mvo_cvar.dot(bl_returns) - risk_free_rate) / np.sqrt(mvo_cvar.values @ covariance.values @ mvo_cvar.values)))
print("Return / volatility: {:.2f}".format(mvo_cvar.dot(bl_returns) / np.sqrt(mvo_cvar.values @ covariance.values @ mvo_cvar.values)))
print()

mvo_weights = pd.DataFrame({
    "Min Volatility": mvo_minvol,
    "Semivariance": mvo_semivariance,
    "CVaR": mvo_cvar,
})


# CVaR check
weight_arr = mvo_cvar.reindex(returns.columns).values
portfolio_rets = (returns * weight_arr).sum(axis=1)
var = portfolio_rets.quantile(0.05)
cvar = portfolio_rets[portfolio_rets <= var].mean()

print("VaR: {:.2f}%".format(100 * var))
print("CVaR: {:.2f}%".format(100 * cvar))

fig1, ax1 = plt.subplots(figsize=(8, 5))
portfolio_rets.hist(bins=50, ax=ax1, color="#4c78a8", edgecolor="white")
ax1.axvline(var, color="red", linestyle="--", linewidth=1.5, label="VaR")
ax1.axvline(cvar, color="black", linestyle="--", linewidth=1.5, label="CVaR")
ax1.set_title("MVO CVaR Return Distribution", fontsize=16, fontweight="bold", pad=12)
ax1.set_xlabel("Daily return")
ax1.set_ylabel("Frequency")
ax1.legend(frameon=False)
ax1.grid(False)

for side in ["top", "right", "bottom", "left"]:
    ax1.spines[side].set_linewidth(1.2)
    ax1.spines[side].set_color("black")

fig1.tight_layout()
plt.show(block=False)


# HRP
import riskfolio as rp

hrp_portfolio = rp.HCPortfolio(returns=returns, w_min=0, w_max=max_weight)
hrp_result = hrp_portfolio.optimization(
    model="HRP",
    codependence="pearson",
    rm="MV",
    rf=0,
    linkage="ward",
    leaf_order=True,
)
hrp_weights = hrp_result["weights"].reindex(assets).fillna(0)
hrp_weights[hrp_weights < 0.0001] = 0
hrp_weights = hrp_weights / hrp_weights.sum()

print("HRP performance:")
print("Expected annual return: {:.2f}%".format(100 * hrp_weights.dot(bl_returns)))
print("Annual volatility: {:.2f}%".format(100 * np.sqrt(hrp_weights.values @ covariance.values @ hrp_weights.values)))
print("Sharpe ratio: {:.2f}".format((hrp_weights.dot(bl_returns) - risk_free_rate) / np.sqrt(hrp_weights.values @ covariance.values @ hrp_weights.values)))
print("Return / volatility: {:.2f}".format(hrp_weights.dot(bl_returns) / np.sqrt(hrp_weights.values @ covariance.values @ hrp_weights.values)))
print()

hrp_cov_weight = covariance.values @ hrp_weights.values
hrp_total_var = hrp_weights.values @ covariance.values @ hrp_weights.values
hrp_risk_contribution = pd.Series(
    hrp_weights.values * hrp_cov_weight / hrp_total_var,
    index=assets,
)

fig_hrp, ax_hrp = plt.subplots(figsize=(10, 5.5))
hrp_risk_contribution.plot(kind="bar", ax=ax_hrp, color="#4c78a8", width=0.72)
ax_hrp.axhline(1 / len(assets), color="red", linewidth=1.5, label="Equal risk")
ax_hrp.set_title("HRP Risk Contribution per Asset", fontsize=16, fontweight="bold", pad=12)
ax_hrp.set_ylabel("Risk contribution")
ax_hrp.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax_hrp.legend(frameon=False)
ax_hrp.grid(False)
ax_hrp.tick_params(axis="x", labelrotation=45, labelsize=10, width=1.1)
ax_hrp.tick_params(axis="y", labelsize=10, width=1.1)

for side in ["top", "right", "bottom", "left"]:
    ax_hrp.spines[side].set_linewidth(1.2)
    ax_hrp.spines[side].set_color("black")

fig_hrp.tight_layout()
plt.show(block=False)


# NCO step 1
correlation = returns.corr()
distance_matrix = np.sqrt(0.5 * (1 - correlation))
distance_matrix = pd.DataFrame(distance_matrix, index=assets, columns=assets)
distance_matrix = (distance_matrix + distance_matrix.T) / 2
np.fill_diagonal(distance_matrix.values, 0)
linkage_matrix = linkage(squareform(distance_matrix), method="ward")
n_clusters = 3
cluster_labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust")

cluster_dict = {}

for i, ticker in enumerate(assets):
    cluster_id = int(cluster_labels[i])

    if cluster_id not in cluster_dict:
        cluster_dict[cluster_id] = []

    cluster_dict[cluster_id].append(ticker)

print()
print("NCO clusters:")
for cluster_id, cluster_assets in cluster_dict.items():
    print("Cluster", cluster_id, ":", cluster_assets)

cluster_order = []

for cluster_id, cluster_assets in cluster_dict.items():
    cluster_order = cluster_order + cluster_assets

cluster_correlation = returns[cluster_order].corr()

fig_cluster, ax_cluster = plt.subplots(figsize=(8, 7))
cluster_plot = ax_cluster.imshow(cluster_correlation.values, cmap="RdYlBu", vmin=-1, vmax=1)

ax_cluster.set_title("NCO Cluster Correlation Map", fontsize=18, fontweight="bold", pad=12)
ax_cluster.set_xticks(np.arange(len(cluster_order)))
ax_cluster.set_yticks(np.arange(len(cluster_order)))
ax_cluster.set_xticklabels(cluster_order, rotation=90)
ax_cluster.set_yticklabels(cluster_order)
ax_cluster.tick_params(axis="both", labelsize=10, length=0)

cluster_start = 0

for cluster_id, cluster_assets in cluster_dict.items():
    cluster_size = len(cluster_assets)
    rect = plt.Rectangle(
        (cluster_start - 0.5, cluster_start - 0.5),
        cluster_size,
        cluster_size,
        fill=False,
        edgecolor="fuchsia",
        linewidth=2.5,
    )
    ax_cluster.add_patch(rect)
    cluster_start = cluster_start + cluster_size

for side in ["top", "right", "bottom", "left"]:
    ax_cluster.spines[side].set_linewidth(1.2)
    ax_cluster.spines[side].set_color("black")

colorbar = fig_cluster.colorbar(cluster_plot, ax=ax_cluster)
colorbar.ax.tick_params(labelsize=9)

fig_cluster.tight_layout()
plt.show(block=False)


# NCO step 2
intra_weights = pd.DataFrame(0.0, index=assets, columns=cluster_dict.keys())

for cluster_id, cluster_assets in cluster_dict.items():
    cluster_cov = covariance.loc[cluster_assets, cluster_assets].values
    n_assets = len(cluster_assets)

    if n_assets == 1:
        weights = np.array([1.0])
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

        weights = result.x

    for i, ticker in enumerate(cluster_assets):
        intra_weights.loc[ticker, cluster_id] = weights[i]


# NCO step 3
cluster_covariance = intra_weights.T @ covariance @ intra_weights
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

inter_weights = inter_result.x
nco_weights = intra_weights.dot(inter_weights)
nco_weights[nco_weights < 0.0001] = 0
nco_weights = nco_weights / nco_weights.sum()

print("NCO performance:")
print("Expected annual return: {:.2f}%".format(100 * nco_weights.dot(bl_returns)))
print("Annual volatility: {:.2f}%".format(100 * np.sqrt(nco_weights.values @ covariance.values @ nco_weights.values)))
print("Sharpe ratio: {:.2f}".format((nco_weights.dot(bl_returns) - risk_free_rate) / np.sqrt(nco_weights.values @ covariance.values @ nco_weights.values)))
print("Return / volatility: {:.2f}".format(nco_weights.dot(bl_returns) / np.sqrt(nco_weights.values @ covariance.values @ nco_weights.values)))
print()


all_weights = pd.DataFrame({
    "BL": bl_allocation,
    "MVO Min Vol": mvo_minvol,
    "MVO Semivariance": mvo_semivariance,
    "MVO CVaR": mvo_cvar,
    "HRP": hrp_weights,
    "NCO": nco_weights,
})

summary = pd.DataFrame(index=all_weights.columns, columns=["return", "volatility", "sharpe", "return_vol_ratio"])

for name in all_weights.columns:
    weight = all_weights[name]
    portfolio_return = weight.dot(bl_returns)
    portfolio_volatility = np.sqrt(weight.values @ covariance.values @ weight.values)

    sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility
    return_vol_ratio = portfolio_return / portfolio_volatility

    summary.loc[name, "return"] = portfolio_return
    summary.loc[name, "volatility"] = portfolio_volatility
    summary.loc[name, "sharpe"] = sharpe
    summary.loc[name, "return_vol_ratio"] = return_vol_ratio

summary = summary.astype(float)

os.makedirs(output_folder, exist_ok=True)
all_weights.to_csv(output_folder + "/optimizer_allocations.csv")
summary.to_csv(output_folder + "/optimizer_summary.csv")

print()
print("Optimizer allocations:")
print(all_weights.round(4).to_string())
print()
print("Optimizer summary:")
print(summary.round(4).to_string())


np.random.seed(2026)
n_samples = 5000
plot_returns = returns.mean() * 252
plot_covariance = returns.cov() * 252
frontier_returns_input = plot_returns
frontier_covariance_input = plot_covariance

random_weights = np.random.dirichlet(np.ones(len(assets)), n_samples)
random_returns = random_weights @ frontier_returns_input.values
random_volatility = np.sqrt(np.diag(random_weights @ frontier_covariance_input.values @ random_weights.T))
random_sharpe = (random_returns - risk_free_rate) / random_volatility

x_low = min(0.15, random_volatility.min() - 0.01)
x_high = max(0.50, random_volatility.max() + 0.04)
y_low = min(0.155, random_returns.min() - 0.01)
y_high = max(0.36, random_returns.max() + 0.04)

for name in all_weights.columns:
    selected_weight = all_weights[name].reindex(assets).fillna(0)
    selected_return = selected_weight.dot(frontier_returns_input)
    selected_volatility = np.sqrt(selected_weight.values @ frontier_covariance_input.values @ selected_weight.values)

    fig_frontier, ax_frontier = plt.subplots(figsize=(7, 5))
    scatter = ax_frontier.scatter(
        random_volatility,
        random_returns,
        c=random_sharpe,
        cmap="viridis",
        s=10,
        alpha=0.85,
        edgecolors="none",
        zorder=1,
    )

    ef_frontier = EfficientFrontier(frontier_returns_input, frontier_covariance_input, weight_bounds=(0, 1))
    plotting.plot_efficient_frontier(
        ef_frontier,
        ax=ax_frontier,
        show_assets=False,
        color="#1f77b4",
        linewidth=1.7,
        label="Efficient frontier",
        zorder=3,
    )
    ax_frontier.scatter(
        selected_volatility,
        selected_return,
        color="red",
        marker="*",
        s=180,
        edgecolor="black",
        linewidth=0.8,
        label=name,
        zorder=4,
    )

    ax_frontier.set_title("Efficient Frontier with random portfolios", fontsize=11, pad=6)
    ax_frontier.set_xlabel("Volatility", fontsize=11)
    ax_frontier.set_ylabel("Return", fontsize=11)
    ax_frontier.legend(frameon=True, fontsize=9, loc="upper left")
    ax_frontier.grid(False)
    ax_frontier.tick_params(axis="both", labelsize=10, width=1.0)
    ax_frontier.set_xlim(x_low, x_high)
    ax_frontier.set_ylim(y_low, y_high)

    for side in ["top", "right", "bottom", "left"]:
        ax_frontier.spines[side].set_linewidth(1.0)
        ax_frontier.spines[side].set_color("black")

    fig_frontier.tight_layout()
    plt.show(block=False)


fig2, ax2 = plt.subplots(figsize=(12, 6))
all_weights.plot(kind="bar", ax=ax2, width=0.75)

ax2.set_title("Optimizer Allocations", fontsize=20, fontweight="bold", pad=12)
ax2.set_ylabel("Weight", fontsize=13)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax2.legend(frameon=False, fontsize=10)
ax2.grid(False)
ax2.tick_params(axis="x", labelrotation=45, labelsize=10, width=1.1)
ax2.tick_params(axis="y", labelsize=10, width=1.1)

for side in ["top", "right", "bottom", "left"]:
    ax2.spines[side].set_linewidth(1.2)
    ax2.spines[side].set_color("black")

fig2.tight_layout()
plt.show(block=False)


fig3, ax3 = plt.subplots(figsize=(9, 5))
summary[["return", "volatility"]].plot(kind="bar", ax=ax3, width=0.7)

ax3.set_title("Optimizer Return and Risk", fontsize=18, fontweight="bold", pad=12)
ax3.set_ylabel("Annual return / volatility", fontsize=12)   
ax3.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax3.legend(["Return", "Volatility"], frameon=False, fontsize=10)
ax3.grid(False)
ax3.tick_params(axis="x", labelrotation=30, labelsize=10, width=1.1)
ax3.tick_params(axis="y", labelsize=10, width=1.1)

for side in ["top", "right", "bottom", "left"]:
    ax3.spines[side].set_linewidth(1.2)
    ax3.spines[side].set_color("black")

fig3.tight_layout()
plt.show(block=False)
