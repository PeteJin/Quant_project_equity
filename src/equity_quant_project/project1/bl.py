import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import yfinance as yf
from pypfopt import black_litterman
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt import EfficientFrontier
from pypfopt import risk_models

try:
    from equity_quant_project.project1 import views
except ModuleNotFoundError:
    import views


assets = views.assets
output_folder = views.output_folder

tau = 0.05
risk_free_rate = 0.04
max_weight = 0.20

prices = views.prices[assets].dropna()
view_table = views.view_table.copy()

market_prices = yf.download(
    "SPY",
    start=views.price_start,
    end=views.price_end,
    auto_adjust=True,
    progress=False,
)["Close"]

S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
delta = black_litterman.market_implied_risk_aversion(market_prices, risk_free_rate=risk_free_rate)

market_caps = {}

for ticker in assets:
    market_caps[ticker] = yf.Ticker(ticker).fast_info["market_cap"]

market_caps = pd.Series(market_caps).reindex(assets)
market_caps = market_caps.replace([np.inf, -np.inf], np.nan)
market_caps = market_caps.fillna(1.0)

prior_return = black_litterman.market_implied_prior_returns(
    market_caps,
    delta,
    S,
    risk_free_rate=risk_free_rate,
)
regime_view = view_table.loc[assets, "forward_1y_regime_view"]

omega = views.omega.loc[assets, assets]

bl = BlackLittermanModel(
    S,
    pi=prior_return,
    absolute_views=regime_view.to_dict(),
    omega=omega,
    tau=tau,
)

bl_return = bl.bl_returns()
bl_covariance = bl.bl_cov()

bl_table = pd.DataFrame({
    "prior_return": prior_return,
    "regime_view": regime_view,
    "bl_return": bl_return,
    "annual_volatility": view_table.loc[assets, "annual_volatility"],
})

bl_table = bl_table.sort_values("bl_return", ascending=False)

ef = EfficientFrontier(bl_return, bl_covariance, weight_bounds=(0, max_weight))
ef.max_sharpe(risk_free_rate=risk_free_rate)
all_weights = pd.Series(ef.clean_weights(), name="weight").reindex(assets).fillna(0)
bl_weights = all_weights[all_weights > 0].sort_values(ascending=False)

expected_return = all_weights.dot(bl_return)
expected_volatility = np.sqrt(all_weights.values @ bl_covariance.values @ all_weights.values)
sharpe_ratio = (expected_return - risk_free_rate) / expected_volatility
return_to_risk = expected_return / expected_volatility

if __name__ == "__main__":
    os.makedirs(output_folder, exist_ok=True)
    bl_table.to_csv(output_folder + "/bl_posterior_returns.csv")
    S.to_csv(output_folder + "/bl_shrinkage_covariance.csv")
    bl_covariance.to_csv(output_folder + "/bl_posterior_covariance.csv")
    prices.to_csv(output_folder + "/project1_prices.csv")
    bl_weights.to_csv(output_folder + "/bl_allocation.csv")

    print("Black-Litterman posterior returns:")
    print(bl_table.round(4).to_string())
    print()
    print("Tau:", tau)
    print("View uncertainty:", views.view_uncertainty)
    print("Risk aversion:", delta)
    print("Market-cap prior weights:")
    print((market_caps / market_caps.sum()).round(4).to_string())
    print()
    print("Output folder:", output_folder)

    fig, ax = plt.subplots(figsize=(11, 6))

    plot_table = bl_table[["prior_return", "bl_return", "regime_view"]]
    plot_table.plot(
        kind="bar",
        ax=ax,
        width=0.75,
        color=["#1f77b4", "#ff7f0e", "#2ca02c"],
    )

    ax.axhline(0, color="black", linewidth=0.9, label="_nolegend_")
    ax.set_title("Black-Litterman Return Views", fontsize=20, fontweight="bold", pad=12)
    ax.set_ylabel("Annual return", fontsize=13)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend(["Prior", "BL Posterior", "Regime View"], frameon=False, fontsize=10)
    ax.grid(False)
    ax.tick_params(axis="x", labelrotation=45, labelsize=10, width=1.1)
    ax.tick_params(axis="y", labelsize=10, width=1.1)

    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_linewidth(1.2)
        ax.spines[side].set_color("black")

    fig.tight_layout()
    plt.show(block=False)

    fig2, ax2 = plt.subplots(figsize=(8, 7))

    cov_limit = abs(bl_covariance.values).max()
    cov_plot = ax2.imshow(bl_covariance.values, cmap="coolwarm", vmin=-cov_limit, vmax=cov_limit, aspect="equal")
    ax2.set_title("Black-Litterman Posterior Covariance Matrix", fontsize=18, fontweight="bold", pad=12)
    ax2.set_xticks(np.arange(len(assets)))
    ax2.set_yticks(np.arange(len(assets)))
    ax2.set_xticklabels(assets, rotation=45, ha="right")
    ax2.set_yticklabels(assets)
    ax2.tick_params(axis="both", labelsize=10, length=0)
    ax2.grid(False)

    for side in ["top", "right", "bottom", "left"]:
        ax2.spines[side].set_linewidth(1.2)
        ax2.spines[side].set_color("black")

    colorbar = fig2.colorbar(cov_plot, ax=ax2)
    colorbar.ax.tick_params(labelsize=9)
    colorbar.set_label("Annual covariance", fontsize=10)

    fig2.tight_layout()
    plt.show(block=False)

    fig3, ax3 = plt.subplots(figsize=(10, 5.5))

    pie_colors = plt.cm.tab20(np.linspace(0, 1, len(bl_weights)))
    ax3.pie(
        bl_weights.values,
        labels=bl_weights.index,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        colors=pie_colors,
        wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
        textprops={"fontsize": 10},
    )

    ax3.set_title("Black-Litterman Allocation", fontsize=20, fontweight="bold", pad=12)
    ax3.axis("equal")

    fig3.tight_layout()
    plt.show(block=False)

    print()
    print("Black-Litterman allocation:")
    print(bl_weights.round(4).to_string())
    print()
    print("Expected annual return:", round(expected_return, 4))
    print("Expected annual volatility:", round(expected_volatility, 4))
    print("Sharpe ratio:", round(sharpe_ratio, 4))
    print("Return / volatility:", round(return_to_risk, 4))
    print("Risk-free rate:", risk_free_rate)
    print("Max weight:", max_weight)
