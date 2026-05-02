import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from hmmlearn.hmm import GaussianHMM


raw_folder = "data/raw/macro"
output_file = "data/processed/project1/hmm_macro_daily_zscore.csv"
start_date = "1997-01-01"


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


hmm_data = pd.concat([vix, spread, curve], axis=1)
hmm_data = hmm_data[hmm_data.index >= start_date]
hmm_data = hmm_data.replace([np.inf, -np.inf], np.nan)
hmm_data = hmm_data.ffill().dropna()

hmm_data_zscore = (hmm_data - hmm_data.mean()) / hmm_data.std(ddof=0)
hmm_data_zscore.index.name = "date"

os.makedirs("data/processed/project1", exist_ok=True)
hmm_data_zscore.to_csv(output_file)


n_states = 2
transition_smoothing = 0.5
variance_scale = 1.5
posterior_sharpness = 1.0
smoothing_window = 126

tightness_raw = (
    hmm_data_zscore["vix"]
    + hmm_data_zscore["high_yield_spread"]
    - hmm_data_zscore["t10y2y"]
)

tightness_index = tightness_raw.rolling(smoothing_window).mean()
tightness_index = tightness_index.dropna()
macro_tightness = (tightness_index - tightness_index.mean()) / tightness_index.std(ddof=0)
hmm_data_zscore = hmm_data_zscore.loc[macro_tightness.index]

state_names = ["Loose", "Tight"]
X = tightness_raw.loc[macro_tightness.index].values.reshape(-1, 1)

starting_state = (macro_tightness >= 0).astype(int)
start_probability = np.array([
    (starting_state == 0).mean(),
    (starting_state == 1).mean(),
])


persistence_penalty = 9000  

transition_matrix_start = pd.DataFrame(transition_smoothing, index=state_names, columns=state_names)

for i in range(len(starting_state) - 1):
    old_state = state_names[starting_state.iloc[i]]
    new_state = state_names[starting_state.iloc[i + 1]]
    transition_matrix_start.loc[old_state, new_state] += 1


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
    n_components=n_states,
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

loose_prob = posterior[:, 0]
tight_prob = posterior[:, 1]

regime_probability = pd.DataFrame(
    posterior,
    index=hmm_data_zscore.index,
    columns=state_names,
)

transition_matrix = pd.DataFrame(
    model.transmat_[state_order][:, state_order],
    index=state_names,
    columns=state_names,
)

fitted_variance = model.covars_
if fitted_variance.ndim == 3:
    fitted_variance = np.array([np.diag(x) for x in fitted_variance])

emission_probability_table = pd.DataFrame(
    {
        "mean": model.means_.ravel()[state_order],
        "variance": fitted_variance.reshape(2, 1).ravel()[state_order],
    },
    index=state_names,
)

regime = posterior.argmax(axis=1)
regime_switches = (regime[1:] != regime[:-1]).sum()

if __name__ == "__main__":
    regime_probability.to_csv("data/processed/project1/hmm_regime_probabilities.csv")
    transition_matrix.to_csv("data/processed/project1/hmm_transition_matrix.csv")
    emission_probability_table.to_csv("data/processed/project1/hmm_emission_table.csv")

    print("Saved:", output_file)
    print("Transition matrix:")
    print(transition_matrix.round(3).to_string())
    print()
    print("Emission probability table:")
    print(emission_probability_table.round(3).to_string())
    print()
    print("Daily observations:", len(hmm_data_zscore))
    print("Regime switches:", regime_switches)
    print("Smoothing window:", smoothing_window)
    print("Posterior sharpness:", posterior_sharpness)

    plt.clf()
    fig1, ax1 = plt.subplots(figsize=(12, 5))

    for column in hmm_data_zscore.columns:
        ax1.plot(hmm_data_zscore.index, hmm_data_zscore[column], linewidth=0.7, label=column)

    ax1.axhline(0, color="gray", linewidth=0.9)
    ax1.set_title("Regime Detection", fontsize=22, fontweight="bold", pad=12)
    ax1.set_ylabel("Z-score", fontsize=14)
    ax1.legend(loc="lower left", frameon=False, fontsize=10)
    ax1.xaxis.set_major_locator(mdates.YearLocator(4))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.set_xlim(pd.Timestamp(start_date), pd.Timestamp("2026-12-31"))
    ax1.grid(False)
    ax1.tick_params(axis="both", labelsize=11, width=1.1)

    for side in ["top", "right", "bottom", "left"]:
        ax1.spines[side].set_linewidth(1.2)
        ax1.spines[side].set_color("black")

    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(12, 4.5))

    ax2.plot(hmm_data_zscore.index, loose_prob, color="green", linewidth=1.0, label="Loose")
    ax2.plot(hmm_data_zscore.index, tight_prob, color="red", linewidth=1.0, linestyle="--", label="Tight")
    ax2.set_title("Regime Posterior Probabilities", fontsize=22, fontweight="bold", pad=12)
    ax2.set_ylabel("Probability", fontsize=14)
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(loc="upper left", frameon=True, fontsize=11)
    ax2.xaxis.set_major_locator(mdates.YearLocator(4))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.set_xlim(pd.Timestamp(start_date), pd.Timestamp("2026-12-31"))
    ax2.grid(False)
    ax2.tick_params(axis="both", labelsize=11, width=1.1)

    for side in ["top", "right", "bottom", "left"]:
        ax2.spines[side].set_linewidth(1.2)
        ax2.spines[side].set_color("black")

    fig2.tight_layout()
    plt.show(block=False)
