import os

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


project1_folder = "data/processed/project1"
output_folder = "data/processed/project3"
table_folder = "reports/tables/project3"
figure_folder = "reports/figures/project3"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(table_folder, exist_ok=True)
os.makedirs(figure_folder, exist_ok=True)

start_date = "2010-01-01"
risk_window = 1890
factor_horizon = 21

price_file = project1_folder + "/project1_prices.csv"

assets = ["CASY", "ORLY", "TLT", "SHY", "LMT", "DE", "MO", "GLD", "MS", "XOM", "EME"]

prices = yf.download(
    assets,
    start=start_date,
    auto_adjust=True,
    threads=False,
    progress=False,
)["Close"]

prices = prices[assets]
prices = prices[prices.index >= start_date]

spy_prices = yf.download(
    ["SPY"],
    start=start_date,
    auto_adjust=True,
    threads=False,
    progress=False,
)["Close"]

if isinstance(spy_prices, pd.DataFrame):
    spy_prices = spy_prices["SPY"]

spy_returns = spy_prices.pct_change(factor_horizon)

hmm_macro_data = pd.read_csv(
    project1_folder + "/hmm_macro_daily_zscore.csv",
    index_col=0,
    parse_dates=True,
)

macro_tightness_index = (
    hmm_macro_data["vix"]
    + hmm_macro_data["high_yield_spread"]
    - hmm_macro_data["t10y2y"]
).rolling(factor_horizon).mean()

high_yield = pd.read_csv("data/raw/macro/High Yield Spread.csv", encoding="latin1")
high_yield.columns = high_yield.columns.str.replace(chr(239) + chr(187) + chr(191), "").str.replace('"', "").str.strip()
high_yield = high_yield[["Date", "Value"]]
high_yield["Date"] = pd.to_datetime(high_yield["Date"], errors="coerce")
high_yield["Value"] = pd.to_numeric(high_yield["Value"], errors="coerce")
high_yield = high_yield.dropna()
high_yield = high_yield.set_index("Date").sort_index()

curve = pd.read_csv("data/raw/macro/T10Y2Y.csv", encoding="latin1")
curve.columns = curve.columns.str.replace(chr(239) + chr(187) + chr(191), "").str.replace('"', "").str.strip()
curve = curve[["observation_date", "T10Y2Y"]]
curve["observation_date"] = pd.to_datetime(curve["observation_date"], errors="coerce")
curve["T10Y2Y"] = pd.to_numeric(curve["T10Y2Y"], errors="coerce")
curve = curve.dropna()
curve = curve.set_index("observation_date").sort_index()

real_yield_file = "data/raw/macro/DFII10.csv"
if os.path.exists(real_yield_file):
    real_yield = pd.read_csv(real_yield_file, encoding="latin1")
else:
    real_yield = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10")

real_yield.columns = real_yield.columns.str.replace(chr(239) + chr(187) + chr(191), "").str.replace('"', "").str.strip()
date_column = "DATE" if "DATE" in real_yield.columns else real_yield.columns[0]
value_column = "DFII10" if "DFII10" in real_yield.columns else real_yield.columns[-1]
real_yield = real_yield[[date_column, value_column]]
real_yield.columns = ["Date", "DFII10"]
real_yield["Date"] = pd.to_datetime(real_yield["Date"], errors="coerce")
real_yield["DFII10"] = pd.to_numeric(real_yield["DFII10"], errors="coerce")
real_yield = real_yield.dropna()
real_yield = real_yield.set_index("Date").sort_index()

factor_raw = pd.DataFrame(index=prices.index)
factor_raw["Downside Market"] = spy_returns.where(spy_returns < 0, 0)
factor_raw["Macro Tightness Shock"] = macro_tightness_index
factor_raw["Curve Shock"] = curve["T10Y2Y"].diff(factor_horizon)
factor_raw["Credit Liquidity Shock"] = high_yield["Value"].diff(factor_horizon)
factor_raw["Real Yield Shock"] = real_yield["DFII10"].diff(factor_horizon)

factor_raw = factor_raw.ffill()
factor_raw = factor_raw[
    [
        "Macro Tightness Shock",
        "Downside Market",
        "Credit Liquidity Shock",
        "Real Yield Shock",
        "Curve Shock",
    ]
]

factor_raw = factor_raw.replace([np.inf, -np.inf], np.nan).dropna()
regime_x = factor_raw["Macro Tightness Shock"].values
regime_X = np.column_stack([np.ones(len(regime_x)), regime_x])

for factor_name in factor_raw.columns:
    if factor_name != "Macro Tightness Shock":
        y = factor_raw[factor_name].values
        coef = np.linalg.lstsq(regime_X, y, rcond=None)[0]
        factor_raw[factor_name] = y - regime_X @ coef

asset_returns = prices.pct_change(factor_horizon)

model_data_full = asset_returns.join(factor_raw, how="inner")
model_data_full = model_data_full.replace([np.inf, -np.inf], np.nan).dropna()
model_data = model_data_full.tail(risk_window)

asset_returns_model = model_data[assets]
factor_raw_model = model_data[factor_raw.columns]

factor_mean = factor_raw_model.mean()
factor_std = factor_raw_model.std(ddof=0)
factor_returns_model = (factor_raw_model - factor_mean) / factor_std
factor_correlation = factor_returns_model.corr()

factor_raw.to_csv(output_folder + "/custom_factor_raw_data.csv")
factor_returns_model.to_csv(output_folder + "/custom_factor_standardized_data.csv")

X = factor_returns_model.values
X = np.column_stack([np.ones(len(X)), X])

beta = pd.DataFrame(index=assets, columns=factor_returns_model.columns, dtype=float)
alpha = pd.Series(index=assets, dtype=float)
specific_variance = pd.Series(index=assets, dtype=float)
residuals = pd.DataFrame(index=asset_returns_model.index, columns=assets, dtype=float)

for ticker in assets:
    y = asset_returns_model[ticker].values
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    fitted = X @ coef
    alpha.loc[ticker] = coef[0] * 252 / factor_horizon
    beta.loc[ticker] = coef[1:]
    residuals[ticker] = y - fitted
    specific_variance.loc[ticker] = residuals[ticker].var() * 252 / factor_horizon

factor_covariance = factor_returns_model.cov() * 252 / factor_horizon
factor_covariance = factor_covariance.loc[beta.columns, beta.columns]

factor_cov_part = beta.values @ factor_covariance.values @ beta.values.T
specific_cov_part = np.diag(specific_variance.values)
factor_model_covariance = pd.DataFrame(
    factor_cov_part + specific_cov_part,
    index=assets,
    columns=assets,
)

sample_covariance = asset_returns_model.cov() * 252 / factor_horizon

weight_file = project1_folder + "/rolling_bl_weight_history.csv"
if os.path.exists(weight_file):
    weights = pd.read_csv(weight_file, index_col=0, parse_dates=True)
    portfolio_weights = weights.iloc[-1].reindex(assets).fillna(0)
else:
    weights = pd.read_csv(project1_folder + "/bl_allocation.csv", index_col=0)
    portfolio_weights = weights.iloc[:, 0].reindex(assets).fillna(0)

portfolio_weights = portfolio_weights / portfolio_weights.sum()

portfolio_factor_exposure = portfolio_weights @ beta

factor_model_variance = float(portfolio_factor_exposure.values @ factor_covariance.values @ portfolio_factor_exposure.values.T)
specific_variance_portfolio = float((portfolio_weights ** 2 * specific_variance).sum())
sample_portfolio_variance = float(portfolio_weights.values @ sample_covariance.values @ portfolio_weights.values)
factor_variance = sample_portfolio_variance - specific_variance_portfolio
factor_variance = max(factor_variance, factor_model_variance)
total_variance = factor_variance + specific_variance_portfolio
portfolio_volatility = np.sqrt(total_variance)

factor_marginal_risk = factor_covariance.values @ portfolio_factor_exposure.values
factor_risk_contribution = portfolio_factor_exposure.values * factor_marginal_risk
factor_risk_contribution = factor_risk_contribution * factor_variance / factor_risk_contribution.sum()
factor_risk_contribution = factor_risk_contribution / total_variance
factor_risk_contribution = pd.Series(factor_risk_contribution, index=factor_covariance.index)

specific_risk_contribution = (portfolio_weights ** 2 * specific_variance) / total_variance

asset_marginal_risk = sample_covariance.values @ portfolio_weights.values
asset_risk_contribution = portfolio_weights.values * asset_marginal_risk / total_variance
asset_risk_contribution = pd.Series(asset_risk_contribution, index=assets)

factor_specific_split = pd.Series(
    {
        "Factor Risk": factor_variance / total_variance,
        "Specific Risk": specific_variance_portfolio / total_variance,
    }
)

portfolio_summary = pd.Series(
    {
        "sample_portfolio_variance": sample_portfolio_variance,
        "raw_factor_model_variance": factor_model_variance,
        "factor_variance": factor_variance,
        "specific_variance": specific_variance_portfolio,
        "total_variance": total_variance,
        "predicted_volatility": portfolio_volatility,
        "factor_risk_share": factor_variance / total_variance,
        "specific_risk_share": specific_variance_portfolio / total_variance,
    }
)

exposure_interpretation = []
for factor_name in beta.columns:
    exposures = beta[factor_name].sort_values()
    exposure_interpretation.append(
        {
            "factor": factor_name,
            "largest_positive_asset": exposures.index[-1],
            "largest_positive_exposure": exposures.iloc[-1],
            "largest_negative_asset": exposures.index[0],
            "largest_negative_exposure": exposures.iloc[0],
        }
    )

exposure_interpretation = pd.DataFrame(exposure_interpretation)

stress_shock = pd.Series(0.0, index=beta.columns)
stress_shock["Macro Tightness Shock"] = 2.0
stress_shock["Credit Liquidity Shock"] = 1.5
stress_shock["Real Yield Shock"] = 1.5

stress_asset_impact = beta @ stress_shock
stress_weighted_impact = portfolio_weights * stress_asset_impact
stress_summary = pd.Series(
    {
        "macro_tightness_shock": stress_shock["Macro Tightness Shock"],
        "credit_liquidity_shock": stress_shock["Credit Liquidity Shock"],
        "real_yield_shock": stress_shock["Real Yield Shock"],
        "portfolio_stress_return": stress_weighted_impact.sum(),
    }
)

stress_table = pd.DataFrame(
    {
        "portfolio_weight": portfolio_weights,
        "asset_stress_return": stress_asset_impact,
        "weighted_stress_contribution": stress_weighted_impact,
    }
)

rolling_window = 756
validation_horizon = 252
validation_step = 21
ewma_half_life = 63
rolling_factor_contribution = []
rolling_factor_specific = []
validation_rows = []

horizon_asset_returns = asset_returns.dropna()
all_weight_dates = weights.index if "weights" in locals() and isinstance(weights.index, pd.DatetimeIndex) else pd.DatetimeIndex([])

rolling_dates = model_data_full.index[rolling_window::validation_step]

for current_date in rolling_dates:
    sample = model_data_full.loc[:current_date].tail(rolling_window)

    if len(sample) < rolling_window:
        continue

    sample_asset_returns = sample[assets]
    sample_factor_raw = sample[factor_raw.columns]
    sample_factor_returns = (sample_factor_raw - sample_factor_raw.mean()) / sample_factor_raw.std(ddof=0)
    sample_X = np.column_stack([np.ones(len(sample_factor_returns)), sample_factor_returns.values])

    ewma_weights = 0.5 ** (np.arange(len(sample_factor_returns) - 1, -1, -1) / ewma_half_life)
    ewma_weights = ewma_weights / ewma_weights.sum()

    factor_weighted_mean = sample_factor_returns.mul(ewma_weights, axis=0).sum()
    factor_centered = sample_factor_returns - factor_weighted_mean
    sample_factor_covariance = factor_centered.mul(ewma_weights, axis=0).T @ factor_centered
    sample_factor_covariance = sample_factor_covariance * 252 / factor_horizon

    asset_weighted_mean = sample_asset_returns.mul(ewma_weights, axis=0).sum()
    asset_centered = sample_asset_returns - asset_weighted_mean
    sample_covariance_rolling = asset_centered.mul(ewma_weights, axis=0).T @ asset_centered
    sample_covariance_rolling = sample_covariance_rolling * 252 / factor_horizon

    sample_beta = pd.DataFrame(index=assets, columns=sample_factor_returns.columns, dtype=float)
    sample_specific = pd.Series(index=assets, dtype=float)

    for ticker in assets:
        y = sample_asset_returns[ticker].values
        coef = np.linalg.lstsq(sample_X, y, rcond=None)[0]
        fitted = sample_X @ coef
        sample_beta.loc[ticker] = coef[1:]
        ticker_residual = y - fitted
        ticker_residual = ticker_residual - np.sum(ewma_weights * ticker_residual)
        sample_specific.loc[ticker] = np.sum(ewma_weights * ticker_residual ** 2) * 252 / factor_horizon

    if len(all_weight_dates) > 0:
        usable_weight_dates = all_weight_dates[all_weight_dates <= current_date]
        if len(usable_weight_dates) > 0:
            current_weight = weights.loc[usable_weight_dates[-1]].reindex(assets).fillna(0)
        else:
            current_weight = portfolio_weights.copy()
    else:
        current_weight = portfolio_weights.copy()

    current_weight = current_weight / current_weight.sum()
    current_factor_exposure = current_weight @ sample_beta
    raw_factor_variance = float(current_factor_exposure.values @ sample_factor_covariance.values @ current_factor_exposure.values.T)
    current_specific_variance = float((current_weight ** 2 * sample_specific).sum())
    current_total_variance = float(current_weight.values @ sample_covariance_rolling.values @ current_weight.values)
    current_factor_variance = max(current_total_variance - current_specific_variance, raw_factor_variance)
    current_total_variance = current_factor_variance + current_specific_variance

    raw_contribution = current_factor_exposure.values * (sample_factor_covariance.values @ current_factor_exposure.values)
    if abs(raw_contribution.sum()) > 0:
        contribution = raw_contribution * current_factor_variance / raw_contribution.sum() / current_total_variance
    else:
        contribution = np.zeros(len(sample_factor_returns.columns))

    row = pd.Series(contribution, index=sample_factor_returns.columns)
    row.name = current_date
    rolling_factor_contribution.append(row)

    split_row = pd.Series(
        {
            "Factor Risk": current_factor_variance / current_total_variance,
            "Specific Risk": current_specific_variance / current_total_variance,
            "Predicted Volatility": np.sqrt(current_total_variance),
        },
        name=current_date,
    )
    rolling_factor_specific.append(split_row)

    realized_returns = horizon_asset_returns[horizon_asset_returns.index <= current_date]
    realized_returns = realized_returns.tail(validation_horizon)
    if len(realized_returns) >= validation_horizon:
        realized_portfolio_returns = realized_returns[assets] @ current_weight
        realized_volatility = realized_portfolio_returns.std() * np.sqrt(252 / factor_horizon)
        validation_rows.append(
            {
                "date": current_date,
                "predicted_volatility": np.sqrt(current_total_variance),
                "realized_volatility": realized_volatility,
                "forecast_error": realized_volatility - np.sqrt(current_total_variance),
            }
        )

rolling_factor_contribution = pd.DataFrame(rolling_factor_contribution)
rolling_factor_specific = pd.DataFrame(rolling_factor_specific)

if len(validation_rows) > 0:
    validation_table = pd.DataFrame(validation_rows).set_index("date")
else:
    validation_table = pd.DataFrame()

if len(validation_table) > 0:
    validation_metrics = pd.Series(
        {
            "mean_predicted_volatility": validation_table["predicted_volatility"].mean(),
            "mean_realized_volatility": validation_table["realized_volatility"].mean(),
            "mean_absolute_error": validation_table["forecast_error"].abs().mean(),
            "rmse": np.sqrt((validation_table["forecast_error"] ** 2).mean()),
            "correlation": validation_table["predicted_volatility"].corr(validation_table["realized_volatility"]),
        }
    )
else:
    validation_metrics = pd.Series(dtype=float)

beta.to_csv(output_folder + "/factor_exposures.csv")
alpha.to_csv(output_folder + "/factor_model_alpha.csv")
factor_covariance.to_csv(output_folder + "/factor_covariance.csv")
factor_correlation.to_csv(output_folder + "/factor_correlation_matrix.csv")
specific_variance.to_csv(output_folder + "/specific_variance.csv")
factor_model_covariance.to_csv(output_folder + "/factor_model_covariance.csv")
portfolio_factor_exposure.to_csv(output_folder + "/portfolio_factor_exposure.csv")
factor_risk_contribution.to_csv(output_folder + "/factor_risk_contribution.csv")
specific_risk_contribution.to_csv(output_folder + "/specific_risk_contribution.csv")
asset_risk_contribution.to_csv(output_folder + "/asset_risk_contribution.csv")
portfolio_summary.to_csv(output_folder + "/portfolio_risk_summary.csv")
factor_specific_split.to_csv(output_folder + "/factor_vs_specific_risk.csv")
exposure_interpretation.to_csv(output_folder + "/asset_factor_exposure_interpretation.csv", index=False)
stress_summary.to_csv(output_folder + "/stress_scenario_summary.csv")
stress_table.to_csv(output_folder + "/stress_scenario_by_asset.csv")
rolling_factor_contribution.to_csv(output_folder + "/rolling_factor_risk_contribution.csv")
rolling_factor_specific.to_csv(output_folder + "/rolling_factor_specific_risk.csv")
validation_table.to_csv(output_folder + "/predicted_vs_realized_volatility.csv")
validation_metrics.to_csv(output_folder + "/validation_metrics.csv")

beta.round(3).to_csv(table_folder + "/factor_exposures.csv")
factor_risk_contribution.round(4).to_csv(table_folder + "/factor_risk_contribution.csv")
asset_risk_contribution.round(4).to_csv(table_folder + "/asset_risk_contribution.csv")
portfolio_summary.round(4).to_csv(table_folder + "/portfolio_risk_summary.csv")
factor_specific_split.round(4).to_csv(table_folder + "/factor_vs_specific_risk.csv")
factor_correlation.round(3).to_csv(table_folder + "/factor_correlation_matrix.csv")
exposure_interpretation.round(4).to_csv(table_folder + "/asset_factor_exposure_interpretation.csv", index=False)
stress_summary.round(4).to_csv(table_folder + "/stress_scenario_summary.csv")
stress_table.round(4).to_csv(table_folder + "/stress_scenario_by_asset.csv")
rolling_factor_contribution.round(4).to_csv(table_folder + "/rolling_factor_risk_contribution.csv")
rolling_factor_specific.round(4).to_csv(table_folder + "/rolling_factor_specific_risk.csv")
validation_table.round(4).to_csv(table_folder + "/predicted_vs_realized_volatility.csv")
validation_metrics.round(4).to_csv(table_folder + "/validation_metrics.csv")

print("\nPortfolio risk summary:")
print(portfolio_summary.round(4))

print("\nLargest factor risk contributions:")
print(factor_risk_contribution.sort_values(ascending=False).round(4))

print("\nLargest asset risk contributions:")
print(asset_risk_contribution.sort_values(ascending=False).round(4))

print("\nValidation metrics:")
print(validation_metrics.round(4))

print("\nStress scenario summary:")
print(stress_summary.round(4))


plt.figure(figsize=(13, 7))
plt.imshow(beta.values, cmap="coolwarm", aspect="auto")
plt.colorbar(label="Factor exposure")
plt.xticks(np.arange(len(beta.columns)), beta.columns, rotation=45, ha="right")
plt.yticks(np.arange(len(beta.index)), beta.index)
plt.title("Barra-Style Factor Exposures", fontsize=22, fontweight="bold")
plt.tight_layout()
plt.savefig(figure_folder + "/Barra-Style Factor Exposures.png", dpi=200)
plt.show()


plt.figure(figsize=(9, 7))
plt.imshow(factor_correlation.values, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(np.arange(len(factor_correlation.columns)), factor_correlation.columns, rotation=45, ha="right")
plt.yticks(np.arange(len(factor_correlation.index)), factor_correlation.index)
plt.title("Factor Correlation Matrix", fontsize=22, fontweight="bold")
plt.tight_layout()
plt.savefig(figure_folder + "/Factor Correlation Matrix.png", dpi=200)
plt.show()


plt.figure(figsize=(11, 6))
plot_data = factor_risk_contribution.sort_values(ascending=False)
plt.bar(plot_data.index, plot_data.values, color="#1f77b4")
plt.axhline(0, color="black", linewidth=1)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.title("Portfolio Factor Risk Contribution", fontsize=22, fontweight="bold")
plt.ylabel("Share of portfolio variance")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(figure_folder + "/Portfolio Factor Risk Contribution.png", dpi=200)
plt.show()


plt.figure(figsize=(11, 6))
asset_plot = asset_risk_contribution.sort_values(ascending=False)
plt.bar(asset_plot.index, asset_plot.values, color="#2ca02c")
plt.axhline(0, color="black", linewidth=1)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.title("Asset Risk Contribution", fontsize=22, fontweight="bold")
plt.ylabel("Share of portfolio variance")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(figure_folder + "/Asset Risk Contribution.png", dpi=200)
plt.show()


if len(rolling_factor_contribution) > 0:
    plt.figure(figsize=(14, 7))
    rolling_plot = rolling_factor_contribution.rolling(3).mean()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    positive_plot = rolling_plot.clip(lower=0)
    negative_plot = rolling_plot.clip(upper=0)
    plt.stackplot(
        rolling_plot.index,
        [positive_plot[column] for column in rolling_plot.columns],
        labels=rolling_plot.columns,
        colors=colors,
        alpha=0.85,
    )
    plt.stackplot(
        rolling_plot.index,
        [negative_plot[column] for column in rolling_plot.columns],
        colors=colors,
        alpha=0.45,
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.title("Rolling Factor Risk Attribution", fontsize=22, fontweight="bold")
    plt.ylabel("Share of portfolio variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_folder + "/Rolling Factor Risk Attribution.png", dpi=200)
    plt.show()


if len(validation_table) > 0:
    plt.figure(figsize=(13, 6))
    plt.plot(validation_table.index, validation_table["predicted_volatility"], linewidth=2.5, label="Model-Implied Volatility")
    plt.plot(validation_table.index, validation_table["realized_volatility"], linewidth=2.5, label="Trailing Realized Volatility")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.title("Predicted vs Realized Volatility", fontsize=22, fontweight="bold")
    plt.ylabel("Annualized volatility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_folder + "/Predicted vs Realized Volatility.png", dpi=200)
    plt.show()


plt.figure(figsize=(11, 6))
stress_plot = stress_table["weighted_stress_contribution"].sort_values()
plt.bar(stress_plot.index, stress_plot.values, color="#d62728")
plt.axhline(0, color="black", linewidth=1)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.title("Stress Scenario by Asset", fontsize=22, fontweight="bold")
plt.ylabel("Weighted return impact")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(figure_folder + "/Stress Scenario by Asset.png", dpi=200)
plt.show()


plt.figure(figsize=(8, 6))
plt.pie(
    factor_specific_split.values,
    labels=factor_specific_split.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=["#1f77b4", "#ff7f0e"],
)
plt.title("Factor Risk vs Specific Risk", fontsize=22, fontweight="bold")
plt.tight_layout()
plt.savefig(figure_folder + "/Factor Risk vs Specific Risk.png", dpi=200)
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(15, 6))

im0 = axes[0].imshow(sample_covariance.values, cmap="coolwarm")
axes[0].set_title("Sample Covariance", fontsize=18, fontweight="bold")
axes[0].set_xticks(np.arange(len(assets)))
axes[0].set_yticks(np.arange(len(assets)))
axes[0].set_xticklabels(assets, rotation=45, ha="right")
axes[0].set_yticklabels(assets)
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(factor_model_covariance.values, cmap="coolwarm")
axes[1].set_title("Factor Model Covariance", fontsize=18, fontweight="bold")
axes[1].set_xticks(np.arange(len(assets)))
axes[1].set_yticks(np.arange(len(assets)))
axes[1].set_xticklabels(assets, rotation=45, ha="right")
axes[1].set_yticklabels(assets)
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(figure_folder + "/Sample vs Factor Model Covariance.png", dpi=200)
plt.show()
