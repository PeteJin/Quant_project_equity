import os

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick

try:
    from equity_quant_project.project1 import hmm
except ModuleNotFoundError:
    import hmm


assets = ["CASY", "ORLY", "TLT", "SHY", "LMT", "DE", "MO", "GLD", "MS", "XOM", "EME"]

output_folder = "data/processed/project1"

price_start = "2005-01-01"
price_end = None

view_horizon_days = 252


regime_probability = hmm.regime_probability.copy()
transition_matrix = hmm.transition_matrix.copy()
state_names = list(regime_probability.columns)
n_states = len(state_names)

current_regime_probability = regime_probability.iloc[-1].values
forward_regime_probability = np.zeros(n_states)

for i in range(1, view_horizon_days + 1):
    forward_regime_probability = forward_regime_probability + current_regime_probability @ np.linalg.matrix_power(transition_matrix.values, i)

forward_regime_probability = forward_regime_probability / view_horizon_days
forward_regime_probability = pd.Series(forward_regime_probability, index=state_names)

os.makedirs(output_folder, exist_ok=True)

prices = yf.download(
    assets,
    start=price_start,
    end=price_end,
    auto_adjust=True,
    threads=False,
    progress=False,
)["Close"]

if isinstance(prices, pd.Series):
    prices = prices.to_frame()

prices = prices.dropna(how="all")

daily_returns = prices.pct_change()
daily_returns = daily_returns.dropna(how="all")
daily_returns = daily_returns.reindex(columns=assets)

cumulative_returns = (1 + daily_returns).cumprod() - 1

forward_daily_returns = daily_returns.shift(-1)
common_dates = forward_daily_returns.index.intersection(regime_probability.index)
regime_daily_returns = forward_daily_returns.loc[common_dates]
regime_probability = regime_probability.loc[common_dates]

regime_daily_returns = regime_daily_returns[regime_daily_returns.index >= pd.Timestamp(price_start)]
regime_daily_returns = regime_daily_returns.dropna(how="all")
regime_probability = regime_probability.loc[regime_daily_returns.index]

regime_state = pd.Series("Loose", index=regime_probability.index)
regime_state[regime_probability["Tight"] > regime_probability["Loose"]] = "Tight"
previous_regime_state = regime_state.shift(1)
regime_threshold = 0.70

loose_days = (
    (previous_regime_state == "Loose")
    & (regime_state == "Loose")
    & (regime_probability["Loose"] >= regime_threshold)
)

tight_days = (
    (previous_regime_state == "Tight")
    & (regime_state == "Tight")
    & (regime_probability["Tight"] >= regime_threshold)
)

loose_weight = regime_probability.loc[loose_days, "Loose"]
tight_weight = regime_probability.loc[tight_days, "Tight"]

loose_return = regime_daily_returns.loc[loose_days].mul(loose_weight, axis=0).sum() / regime_daily_returns.loc[loose_days].notna().mul(loose_weight, axis=0).sum()
tight_return = regime_daily_returns.loc[tight_days].mul(tight_weight, axis=0).sum() / regime_daily_returns.loc[tight_days].notna().mul(tight_weight, axis=0).sum()

regime_return_table = pd.DataFrame({
    "Loose": loose_return,
    "Tight": tight_return,
})
regime_return_table["Spread"] = regime_return_table["Loose"] - regime_return_table["Tight"]

daily_view = (
    regime_return_table["Loose"] * forward_regime_probability["Loose"]
    + regime_return_table["Tight"] * forward_regime_probability["Tight"]
)

annual_view = daily_view * 252

annual_vol = daily_returns.std() * np.sqrt(252)
regime_probability_gap = abs(forward_regime_probability["Loose"] - forward_regime_probability["Tight"])
view_uncertainty = 0.25

view_table = pd.DataFrame({
    "loose_daily_return": regime_return_table["Loose"],
    "tight_daily_return": regime_return_table["Tight"],
    "loose_tight_spread": regime_return_table["Spread"],
    "forward_1y_regime_view": annual_view,
    "annual_volatility": annual_vol,
    "regime_probability_gap": regime_probability_gap,
})

regime_return_plot = pd.DataFrame({
    "Loose": regime_return_table["Loose"] * 252,
    "Tight": regime_return_table["Tight"] * 252,
})

pick_matrix = pd.DataFrame(
    np.eye(len(assets)),
    index=assets,
    columns=assets,
)

omega = pd.DataFrame(
    np.diag((annual_vol * view_uncertainty) ** 2),
    index=assets,
    columns=assets,
)

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(11, 7))
    colors = plt.cm.tab20(np.linspace(0, 1, len(assets)))

    for i, ticker in enumerate(assets):
        ax.plot(
            cumulative_returns.index,
            cumulative_returns[ticker],
            linewidth=1.8,
            color=colors[i],
            label=ticker,
        )

    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_title("Cumulative Stock Returns", fontsize=22, fontweight="bold", pad=12)
    ax.set_ylabel("Cumulative return", fontsize=14)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=10)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(pd.Timestamp(price_start), pd.Timestamp("2026-12-31"))
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=11, width=1.1)

    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_linewidth(1.2)
        ax.spines[side].set_color("black")

    fig.tight_layout()
    plt.show(block=False)

    fig2, ax2 = plt.subplots(figsize=(11, 6))
    regime_return_plot.plot(kind="bar", ax=ax2, width=0.75, color=["green", "red"])

    ax2.axhline(0, color="black", linewidth=0.9, label="_nolegend_")
    ax2.set_title("Regime-Conditioned Stock Returns", fontsize=20, fontweight="bold", pad=12)
    ax2.set_ylabel("Annualized return", fontsize=13)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.legend(["Loose", "Tight"], frameon=False, fontsize=10)
    ax2.grid(False)
    ax2.tick_params(axis="x", labelrotation=45, labelsize=10, width=1.1)
    ax2.tick_params(axis="y", labelsize=10, width=1.1)

    for side in ["top", "right", "bottom", "left"]:
        ax2.spines[side].set_linewidth(1.2)
        ax2.spines[side].set_color("black")

    fig2.tight_layout()
    plt.show(block=False)

    forward_regime_probability.to_csv(output_folder + "/hmm_forward_252d_regime_probability.csv")
    view_table.to_csv(output_folder + "/bl_return_views.csv")
    pick_matrix.to_csv(output_folder + "/bl_pick_matrix.csv")
    omega.to_csv(output_folder + "/bl_omega.csv")

    print("Current regime probability:")
    print(pd.Series(current_regime_probability, index=state_names).round(3).to_string())
    print()
    print("Forward 252-day regime probability:")
    print(forward_regime_probability.round(3).to_string())
    print()
    print("Regime return sample days:")
    print("Loose:", int(loose_days.sum()))
    print("Tight:", int(tight_days.sum()))
    print()
    print("Black-Litterman return views:")
    print(view_table.round(4).to_string())
    print()
    print("Saved BL input files to:", output_folder)
