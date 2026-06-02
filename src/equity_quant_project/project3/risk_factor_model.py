import os

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.covariance import LedoitWolf


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
ewma_half_life = 63

assets = ["CASY", "ORLY", "TLT", "SHY", "LMT", "DE", "MO", "GLD", "MS", "XOM", "EME"]
base_factor_tickers = ["SPY", "TIP", "HYG"]

prices = yf.download(
    assets,
    start=start_date,
    auto_adjust=True,
    threads=False,
    progress=False,
)["Close"]

prices = prices[assets]
prices = prices[prices.index >= start_date]

base_prices = yf.download(
    base_factor_tickers,
    start=start_date,
    auto_adjust=True,
    threads=False,
    progress=False,
)["Close"]

base_prices = base_prices[base_factor_tickers]

hmm_regime_data = pd.read_csv(
    project1_folder + "/hmm_regime_probabilities.csv",
    index_col=0,
    parse_dates=True,
)
hmm_last_date = hmm_regime_data.index.max()
prices = prices[prices.index <= hmm_last_date]
base_prices = base_prices[base_prices.index <= hmm_last_date]

transition_matrix = pd.read_csv(
    project1_folder + "/hmm_transition_matrix.csv",
    index_col=0,
)
transition_matrix = transition_matrix.loc[["Loose", "Tight"], ["Loose", "Tight"]]
transition_horizon = np.linalg.matrix_power(transition_matrix.values, factor_horizon)

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

raw_folder = "data/raw/macro"

vix = pd.read_csv(raw_folder + "/VIX.csv", encoding="latin1")
vix.columns = vix.columns.str.replace(chr(239) + chr(187) + chr(191), "").str.replace('"', "").str.strip()
vix = vix[["DATE", "CLOSE"]]
vix["DATE"] = pd.to_datetime(vix["DATE"], errors="coerce")
vix["CLOSE"] = pd.to_numeric(vix["CLOSE"], errors="coerce")
vix = vix.dropna()
vix = vix.set_index("DATE").sort_index()
vix = vix.pct_change()
vix.columns = ["VIX Tightness Shock"]

spread = pd.read_csv(raw_folder + "/High Yield Spread.csv", encoding="latin1")
spread.columns = spread.columns.str.replace(chr(239) + chr(187) + chr(191), "").str.replace('"', "").str.strip()
spread = spread[["Date", "Value"]]
spread["Date"] = pd.to_datetime(spread["Date"], errors="coerce")
spread["Value"] = pd.to_numeric(spread["Value"], errors="coerce")
spread = spread.dropna()
spread = spread.set_index("Date").sort_index()
spread = spread.diff()
spread.columns = ["HY Spread Tightness Shock"]

curve = pd.read_csv(raw_folder + "/T10Y2Y.csv", encoding="latin1")
curve.columns = curve.columns.str.replace(chr(239) + chr(187) + chr(191), "").str.replace('"', "").str.strip()
curve = curve[["observation_date", "T10Y2Y"]]
curve["observation_date"] = pd.to_datetime(curve["observation_date"], errors="coerce")
curve["T10Y2Y"] = pd.to_numeric(curve["T10Y2Y"], errors="coerce")
curve = curve.dropna()
curve = curve.set_index("observation_date").sort_index()
curve = -curve.diff()
curve.columns = ["T10Y2Y Tightness Shock"]

daily_asset_returns = prices.pct_change().dropna()
asset_returns = prices.pct_change(factor_horizon).shift(-factor_horizon)
base_factor_raw = base_prices.pct_change(factor_horizon)
base_factor_raw.columns = ["Equity Market Beta", "TIPS Duration Beta", "Credit Market Beta"]

regime_probability = hmm_regime_data[["Loose", "Tight"]].reindex(prices.index).ffill()
regime_state = pd.Series("Loose", index=regime_probability.index)
regime_state[regime_probability["Tight"] > regime_probability["Loose"]] = "Tight"
previous_regime_state = regime_state.shift(1)
regime_threshold = 0.70

loose_regime = (
    (previous_regime_state == "Loose")
    & (regime_state == "Loose")
    & (regime_probability["Loose"] >= regime_threshold)
)

tight_regime = (
    (previous_regime_state == "Tight")
    & (regime_state == "Tight")
    & (regime_probability["Tight"] >= regime_threshold)
)

return_regime_probability = regime_probability
return_loose_regime = loose_regime.fillna(False)
return_tight_regime = tight_regime.fillna(False)

regime_factor_daily = pd.concat([vix, spread, curve], axis=1)
regime_factor_daily = regime_factor_daily.replace([np.inf, -np.inf], np.nan)
regime_factor_daily = regime_factor_daily.ffill().dropna()

regime_factor_raw = regime_factor_daily.rolling(factor_horizon).sum()
regime_factor_raw = regime_factor_raw.reindex(prices.index).ffill()

custom_factor_raw = pd.DataFrame(index=prices.index)
custom_factor_raw["Downside Market Shock"] = base_factor_raw["Equity Market Beta"].where(base_factor_raw["Equity Market Beta"] < 0, 0)
custom_factor_raw["Real Yield Shock"] = real_yield["DFII10"].diff(factor_horizon)

factor_raw = regime_factor_raw.join(base_factor_raw, how="outer").join(custom_factor_raw, how="outer").ffill()
model_data_full = asset_returns.join(factor_raw, how="inner")
model_data_full = model_data_full.replace([np.inf, -np.inf], np.nan).dropna()
model_data = model_data_full.tail(risk_window)

asset_returns_model = model_data[assets]
regime_factor_model = model_data[regime_factor_raw.columns]
base_factor_model = model_data[base_factor_raw.columns]
custom_factor_model = model_data[custom_factor_raw.columns]

ewma_weights = 0.5 ** (np.arange(len(model_data) - 1, -1, -1) / ewma_half_life)
ewma_weights = ewma_weights / ewma_weights.sum()

regime_mean = regime_factor_model.mul(ewma_weights, axis=0).sum()
regime_centered = regime_factor_model - regime_mean
regime_std = np.sqrt((regime_centered ** 2).mul(ewma_weights, axis=0).sum())
regime_factor_z = regime_centered / regime_std

base_mean = base_factor_model.mul(ewma_weights, axis=0).sum()
base_centered = base_factor_model - base_mean
base_std = np.sqrt((base_centered ** 2).mul(ewma_weights, axis=0).sum())
base_factor_z = base_centered / base_std

custom_mean = custom_factor_model.mul(ewma_weights, axis=0).sum()
custom_centered = custom_factor_model - custom_mean
custom_std = np.sqrt((custom_centered ** 2).mul(ewma_weights, axis=0).sum())
custom_factor_z = custom_centered / custom_std

sqrt_w = np.sqrt(ewma_weights)

regime_factor_names = list(regime_factor_raw.columns)
base_factor_names = list(base_factor_raw.columns)
custom_factor_names = list(custom_factor_raw.columns)

regime_anchor = regime_factor_z[regime_factor_names]

regime_X = np.column_stack([np.ones(len(regime_anchor)), regime_anchor.values])
weighted_regime_X = regime_X * sqrt_w[:, None]

purged_base_factor = pd.DataFrame(index=base_factor_z.index, columns=base_factor_z.columns, dtype=float)
base_purge_beta = pd.DataFrame(index=base_factor_z.columns, columns=["Intercept"] + regime_factor_names, dtype=float)

for factor_name in base_factor_z.columns:
    y = base_factor_z[factor_name].values
    coef = np.linalg.lstsq(weighted_regime_X, y * sqrt_w, rcond=None)[0]
    base_purge_beta.loc[factor_name] = coef
    purged_base_factor[factor_name] = y - regime_X @ coef

purged_base_mean = purged_base_factor.mul(ewma_weights, axis=0).sum()
purged_base_centered = purged_base_factor - purged_base_mean
purged_base_std = np.sqrt((purged_base_centered ** 2).mul(ewma_weights, axis=0).sum())
purged_base_factor = purged_base_centered / purged_base_std

purge_driver = regime_anchor.join(purged_base_factor)
purge_X = np.column_stack([np.ones(len(purge_driver)), purge_driver.values])
weighted_purge_X = purge_X * sqrt_w[:, None]

other_custom_factors = list(custom_factor_z.columns)
purge_beta = pd.DataFrame(index=other_custom_factors, columns=["Intercept"] + list(purge_driver.columns), dtype=float)
purged_custom_factor = pd.DataFrame(index=custom_factor_z.index, columns=other_custom_factors, dtype=float)

for factor_name in other_custom_factors:
    y = custom_factor_z[factor_name].values
    coef = np.linalg.lstsq(weighted_purge_X, y * sqrt_w, rcond=None)[0]
    purge_beta.loc[factor_name] = coef
    purged_custom_factor[factor_name] = y - purge_X @ coef

purged_mean = purged_custom_factor.mul(ewma_weights, axis=0).sum()
purged_centered = purged_custom_factor - purged_mean
purged_std = np.sqrt((purged_centered ** 2).mul(ewma_weights, axis=0).sum())
purged_custom_factor = purged_centered / purged_std

factor_returns_model = regime_anchor.join(purged_base_factor).join(purged_custom_factor)
factor_correlation = factor_returns_model.corr()

factor_raw.to_csv(output_folder + "/factor_raw_data.csv")
regime_factor_raw.to_csv(output_folder + "/regime_factor_raw_data.csv")
custom_factor_raw.to_csv(output_folder + "/custom_factor_raw_data.csv")
regime_factor_z.to_csv(output_folder + "/regime_factor_standardized_data.csv")
base_factor_z.to_csv(output_folder + "/base_factor_standardized_data.csv")
custom_factor_z.to_csv(output_folder + "/custom_factor_standardized_data.csv")
purged_base_factor.to_csv(output_folder + "/purged_base_factors.csv")
base_purge_beta.to_csv(output_folder + "/base_factor_purge_beta.csv")
purged_custom_factor.to_csv(output_folder + "/purged_custom_macro_shocks.csv")
purge_beta.to_csv(output_folder + "/custom_factor_purge_beta.csv")

regime_probability_model = return_regime_probability.reindex(model_data.index).ffill()
loose_regime_model = return_loose_regime.reindex(model_data.index).fillna(False)
tight_regime_model = return_tight_regime.reindex(model_data.index).fillna(False)
base_observation_weight = pd.Series(ewma_weights, index=model_data.index)

loose_observation_weight = base_observation_weight * regime_probability_model["Loose"] * loose_regime_model.astype(float)
tight_observation_weight = base_observation_weight * regime_probability_model["Tight"] * tight_regime_model.astype(float)

if loose_observation_weight.sum() == 0:
    loose_observation_weight = base_observation_weight * regime_probability_model["Loose"]
if tight_observation_weight.sum() == 0:
    tight_observation_weight = base_observation_weight * regime_probability_model["Tight"]

loose_observation_weight = loose_observation_weight / loose_observation_weight.sum()
tight_observation_weight = tight_observation_weight / tight_observation_weight.sum()

factor_names = list(factor_returns_model.columns)
X_factor = np.column_stack([np.ones(len(factor_returns_model)), factor_returns_model.values])
weighted_X_factor = X_factor * sqrt_w[:, None]

beta = pd.DataFrame(index=assets, columns=factor_returns_model.columns, dtype=float)
alpha = pd.Series(index=assets, dtype=float)
residuals = pd.DataFrame(index=asset_returns_model.index, columns=assets, dtype=float)
specific_variance = pd.Series(index=assets, dtype=float)

for ticker in assets:
    y = asset_returns_model[ticker].values
    coef = np.linalg.lstsq(weighted_X_factor, y * sqrt_w, rcond=None)[0]
    fitted = X_factor @ coef
    ticker_residual = y - fitted
    ticker_residual = ticker_residual - np.sum(ewma_weights * ticker_residual)

    alpha.loc[ticker] = coef[0] * 252 / factor_horizon
    beta.loc[ticker, factor_names] = coef[1:]
    residuals[ticker] = ticker_residual
    specific_variance.loc[ticker] = np.sum(ewma_weights * ticker_residual ** 2) * 252 / factor_horizon

loose_factor_mean = factor_returns_model.mul(loose_observation_weight, axis=0).sum()
loose_factor_centered = factor_returns_model - loose_factor_mean
loose_factor_weighted = loose_factor_centered.values * np.sqrt(loose_observation_weight.values * len(loose_factor_centered))[:, None]
loose_factor_model = LedoitWolf(assume_centered=True)
loose_factor_model.fit(loose_factor_weighted)
loose_factor_covariance = pd.DataFrame(
    loose_factor_model.covariance_ * 252 / factor_horizon,
    index=factor_returns_model.columns,
    columns=factor_returns_model.columns,
)

tight_factor_mean = factor_returns_model.mul(tight_observation_weight, axis=0).sum()
tight_factor_centered = factor_returns_model - tight_factor_mean
tight_factor_weighted = tight_factor_centered.values * np.sqrt(tight_observation_weight.values * len(tight_factor_centered))[:, None]
tight_factor_model = LedoitWolf(assume_centered=True)
tight_factor_model.fit(tight_factor_weighted)
tight_factor_covariance = pd.DataFrame(
    tight_factor_model.covariance_ * 252 / factor_horizon,
    index=factor_returns_model.columns,
    columns=factor_returns_model.columns,
)

current_regime_probability = pd.Series(
    regime_probability.iloc[-1].loc[["Loose", "Tight"]].values @ transition_horizon,
    index=["Loose", "Tight"],
)
factor_covariance = (
    loose_factor_covariance * current_regime_probability["Loose"]
    + tight_factor_covariance * current_regime_probability["Tight"]
)

loose_residual_mean = residuals.mul(loose_observation_weight, axis=0).sum()
loose_residual_centered = residuals - loose_residual_mean
loose_residual_weighted = loose_residual_centered.values * np.sqrt(loose_observation_weight.values * len(loose_residual_centered))[:, None]
loose_specific_model = LedoitWolf(assume_centered=True)
loose_specific_model.fit(loose_residual_weighted)
loose_specific_covariance = pd.DataFrame(
    loose_specific_model.covariance_ * 252 / factor_horizon,
    index=assets,
    columns=assets,
)

tight_residual_mean = residuals.mul(tight_observation_weight, axis=0).sum()
tight_residual_centered = residuals - tight_residual_mean
tight_residual_weighted = tight_residual_centered.values * np.sqrt(tight_observation_weight.values * len(tight_residual_centered))[:, None]
tight_specific_model = LedoitWolf(assume_centered=True)
tight_specific_model.fit(tight_residual_weighted)
tight_specific_covariance = pd.DataFrame(
    tight_specific_model.covariance_ * 252 / factor_horizon,
    index=assets,
    columns=assets,
)

specific_covariance = (
    loose_specific_covariance * current_regime_probability["Loose"]
    + tight_specific_covariance * current_regime_probability["Tight"]
)
specific_covariance = pd.DataFrame(
    np.diag(np.diag(specific_covariance.values)),
    index=assets,
    columns=assets,
)
specific_shrinkage = (
    loose_specific_model.shrinkage_ * current_regime_probability["Loose"]
    + tight_specific_model.shrinkage_ * current_regime_probability["Tight"]
)

factor_cov_part = beta.values @ factor_covariance.values @ beta.values.T
factor_cov_part = pd.DataFrame(factor_cov_part, index=assets, columns=assets)
factor_model_covariance = factor_cov_part + specific_covariance

asset_centered = asset_returns_model - asset_returns_model.mul(ewma_weights, axis=0).sum()
sample_covariance = asset_centered.mul(ewma_weights, axis=0).T @ asset_centered
sample_covariance = sample_covariance * 252 / factor_horizon

weight_file = project1_folder + "/rolling_bl_weight_history.csv"
if os.path.exists(weight_file):
    weights = pd.read_csv(weight_file, index_col=0, parse_dates=True)
    portfolio_weights = weights.iloc[-1].reindex(assets).fillna(0)
else:
    weights = pd.read_csv(project1_folder + "/bl_allocation.csv", index_col=0)
    portfolio_weights = weights.iloc[:, 0].reindex(assets).fillna(0)

portfolio_weights = portfolio_weights / portfolio_weights.sum()

portfolio_factor_exposure = portfolio_weights @ beta
linear_factor_variance = float(portfolio_factor_exposure.values @ factor_covariance.values @ portfolio_factor_exposure.values.T)
factor_variance = linear_factor_variance
specific_variance_portfolio = float(portfolio_weights.values @ specific_covariance.values @ portfolio_weights.values)
total_variance = factor_variance + specific_variance_portfolio
portfolio_volatility = np.sqrt(total_variance)

factor_marginal_risk = factor_covariance.values @ portfolio_factor_exposure.values
factor_risk_contribution = portfolio_factor_exposure.values * factor_marginal_risk / total_variance
factor_risk_contribution = pd.Series(factor_risk_contribution, index=factor_covariance.index)

specific_marginal_risk = specific_covariance.values @ portfolio_weights.values
specific_risk_contribution = portfolio_weights.values * specific_marginal_risk / total_variance
specific_risk_contribution = pd.Series(specific_risk_contribution, index=assets)

asset_marginal_risk = factor_model_covariance.values @ portfolio_weights.values
asset_risk_contribution = portfolio_weights.values * asset_marginal_risk / total_variance
asset_risk_contribution = pd.Series(asset_risk_contribution, index=assets)

factor_specific_split = pd.Series(
    {
        "Systematic Risk": factor_variance / total_variance,
        "Specific Risk": specific_variance_portfolio / total_variance,
    }
)

factor_group_contribution = pd.Series(
    {
        "Regime Tightness Risk": factor_risk_contribution.loc[regime_factor_names].sum(),
        "Equity Market Beta": factor_risk_contribution.loc["Equity Market Beta"],
        "TIPS Duration Beta": factor_risk_contribution.loc["TIPS Duration Beta"],
        "Credit Market Beta": factor_risk_contribution.loc["Credit Market Beta"],
        "Downside Market Shock": factor_risk_contribution.loc["Downside Market Shock"],
        "Real Yield Shock": factor_risk_contribution.loc["Real Yield Shock"],
        "Specific Risk": specific_variance_portfolio / total_variance,
    }
)

portfolio_summary = pd.Series(
    {
        "linear_factor_variance": linear_factor_variance,
        "factor_variance": factor_variance,
        "specific_variance": specific_variance_portfolio,
        "total_variance": total_variance,
        "predicted_volatility": portfolio_volatility,
        "factor_risk_share": factor_variance / total_variance,
        "specific_risk_share": specific_variance_portfolio / total_variance,
        "specific_shrinkage": specific_shrinkage,
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
stress_shock["VIX Tightness Shock"] = 2.0
stress_shock["HY Spread Tightness Shock"] = 2.0
stress_shock["T10Y2Y Tightness Shock"] = 2.0
stress_shock["Real Yield Shock"] = 1.5
stress_shock["Credit Market Beta"] = -1.0
stress_shock["TIPS Duration Beta"] = -1.0

stress_asset_impact = beta @ stress_shock
stress_weighted_impact = portfolio_weights * stress_asset_impact
stress_summary = pd.Series(
    {
        "vix_tightness_shock": stress_shock["VIX Tightness Shock"],
        "hy_spread_tightness_shock": stress_shock["HY Spread Tightness Shock"],
        "t10y2y_tightness_shock": stress_shock["T10Y2Y Tightness Shock"],
        "real_yield_shock": stress_shock["Real Yield Shock"],
        "credit_market_beta": stress_shock["Credit Market Beta"],
        "tips_duration_beta": stress_shock["TIPS Duration Beta"],
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
validation_step = 21
rolling_factor_contribution = []
rolling_factor_specific = []
validation_rows = []
all_weight_dates = weights.index if "weights" in locals() and isinstance(weights.index, pd.DatetimeIndex) else pd.DatetimeIndex([])
rolling_dates = model_data_full.index[rolling_window::validation_step]

for current_date in rolling_dates:
    sample = model_data_full.loc[:current_date].tail(rolling_window)

    if len(sample) < rolling_window:
        continue

    sample_assets = sample[assets]
    sample_regime = sample[regime_factor_raw.columns]
    sample_base = sample[base_factor_raw.columns]
    sample_custom = sample[custom_factor_raw.columns]

    sample_weights = 0.5 ** (np.arange(len(sample) - 1, -1, -1) / ewma_half_life)
    sample_weights = sample_weights / sample_weights.sum()
    sample_sqrt_w = np.sqrt(sample_weights)

    sample_regime_mean = sample_regime.mul(sample_weights, axis=0).sum()
    sample_regime_centered = sample_regime - sample_regime_mean
    sample_regime_std = np.sqrt((sample_regime_centered ** 2).mul(sample_weights, axis=0).sum())
    sample_regime_z = sample_regime_centered / sample_regime_std

    sample_base_mean = sample_base.mul(sample_weights, axis=0).sum()
    sample_base_centered = sample_base - sample_base_mean
    sample_base_std = np.sqrt((sample_base_centered ** 2).mul(sample_weights, axis=0).sum())
    sample_base_z = sample_base_centered / sample_base_std

    sample_custom_mean = sample_custom.mul(sample_weights, axis=0).sum()
    sample_custom_centered = sample_custom - sample_custom_mean
    sample_custom_std = np.sqrt((sample_custom_centered ** 2).mul(sample_weights, axis=0).sum())
    sample_custom_z = sample_custom_centered / sample_custom_std

    sample_regime_anchor = sample_regime_z[regime_factor_names]
    sample_regime_X = np.column_stack([np.ones(len(sample_regime_anchor)), sample_regime_anchor.values])
    sample_weighted_regime_X = sample_regime_X * sample_sqrt_w[:, None]

    sample_purged_base = pd.DataFrame(index=sample_base_z.index, columns=sample_base_z.columns, dtype=float)

    for factor_name in sample_base_z.columns:
        y = sample_base_z[factor_name].values
        coef = np.linalg.lstsq(sample_weighted_regime_X, y * sample_sqrt_w, rcond=None)[0]
        sample_purged_base[factor_name] = y - sample_regime_X @ coef

    sample_purged_base_mean = sample_purged_base.mul(sample_weights, axis=0).sum()
    sample_purged_base_centered = sample_purged_base - sample_purged_base_mean
    sample_purged_base_std = np.sqrt((sample_purged_base_centered ** 2).mul(sample_weights, axis=0).sum())
    sample_purged_base = sample_purged_base_centered / sample_purged_base_std

    sample_purge_driver = sample_regime_anchor.join(sample_purged_base)
    sample_purge_X = np.column_stack([np.ones(len(sample_purge_driver)), sample_purge_driver.values])
    sample_weighted_purge_X = sample_purge_X * sample_sqrt_w[:, None]
    sample_purged = pd.DataFrame(index=sample_custom_z.index, columns=other_custom_factors, dtype=float)

    for factor_name in other_custom_factors:
        y = sample_custom_z[factor_name].values
        coef = np.linalg.lstsq(sample_weighted_purge_X, y * sample_sqrt_w, rcond=None)[0]
        sample_purged[factor_name] = y - sample_purge_X @ coef

    sample_purged_mean = sample_purged.mul(sample_weights, axis=0).sum()
    sample_purged_centered = sample_purged - sample_purged_mean
    sample_purged_std = np.sqrt((sample_purged_centered ** 2).mul(sample_weights, axis=0).sum())
    sample_purged = sample_purged_centered / sample_purged_std

    sample_factors = sample_regime_anchor.join(sample_purged_base).join(sample_purged)
    sample_factor_names = list(sample_factors.columns)
    sample_X_factor = np.column_stack([np.ones(len(sample_factors)), sample_factors.values])
    sample_weighted_X_factor = sample_X_factor * sample_sqrt_w[:, None]

    sample_regime_probability = return_regime_probability.reindex(sample.index).ffill()
    sample_loose_regime = return_loose_regime.reindex(sample.index).fillna(False)
    sample_tight_regime = return_tight_regime.reindex(sample.index).fillna(False)
    sample_weight_series = pd.Series(sample_weights, index=sample.index)

    sample_loose_weight = sample_weight_series * sample_regime_probability["Loose"] * sample_loose_regime.astype(float)
    sample_tight_weight = sample_weight_series * sample_regime_probability["Tight"] * sample_tight_regime.astype(float)

    if sample_loose_weight.sum() == 0:
        sample_loose_weight = sample_weight_series * sample_regime_probability["Loose"]
    if sample_tight_weight.sum() == 0:
        sample_tight_weight = sample_weight_series * sample_regime_probability["Tight"]

    sample_loose_weight = sample_loose_weight / sample_loose_weight.sum()
    sample_tight_weight = sample_tight_weight / sample_tight_weight.sum()
    sample_beta = pd.DataFrame(index=assets, columns=sample_factors.columns, dtype=float)
    sample_residuals = pd.DataFrame(index=sample_assets.index, columns=assets, dtype=float)

    for ticker in assets:
        y = sample_assets[ticker].values
        coef = np.linalg.lstsq(sample_weighted_X_factor, y * sample_sqrt_w, rcond=None)[0]
        fitted = sample_X_factor @ coef
        ticker_residual = y - fitted
        ticker_residual = ticker_residual - np.sum(sample_weights * ticker_residual)
        sample_beta.loc[ticker, sample_factor_names] = coef[1:]
        sample_residuals[ticker] = ticker_residual

    sample_loose_factor_mean = sample_factors.mul(sample_loose_weight, axis=0).sum()
    sample_loose_factor_centered = sample_factors - sample_loose_factor_mean
    sample_loose_factor_weighted = sample_loose_factor_centered.values * np.sqrt(sample_loose_weight.values * len(sample_loose_factor_centered))[:, None]
    sample_loose_factor_model = LedoitWolf(assume_centered=True)
    sample_loose_factor_model.fit(sample_loose_factor_weighted)
    sample_loose_factor_covariance = pd.DataFrame(
        sample_loose_factor_model.covariance_ * 252 / factor_horizon,
        index=sample_factors.columns,
        columns=sample_factors.columns,
    )

    sample_tight_factor_mean = sample_factors.mul(sample_tight_weight, axis=0).sum()
    sample_tight_factor_centered = sample_factors - sample_tight_factor_mean
    sample_tight_factor_weighted = sample_tight_factor_centered.values * np.sqrt(sample_tight_weight.values * len(sample_tight_factor_centered))[:, None]
    sample_tight_factor_model = LedoitWolf(assume_centered=True)
    sample_tight_factor_model.fit(sample_tight_factor_weighted)
    sample_tight_factor_covariance = pd.DataFrame(
        sample_tight_factor_model.covariance_ * 252 / factor_horizon,
        index=sample_factors.columns,
        columns=sample_factors.columns,
    )

    sample_current_probability = pd.Series(
        sample_regime_probability.iloc[-1].loc[["Loose", "Tight"]].values @ transition_horizon,
        index=["Loose", "Tight"],
    )
    sample_factor_covariance = (
        sample_loose_factor_covariance * sample_current_probability["Loose"]
        + sample_tight_factor_covariance * sample_current_probability["Tight"]
    )

    sample_loose_residual_mean = sample_residuals.mul(sample_loose_weight, axis=0).sum()
    sample_loose_residual_centered = sample_residuals - sample_loose_residual_mean
    sample_loose_residual_weighted = sample_loose_residual_centered.values * np.sqrt(sample_loose_weight.values * len(sample_loose_residual_centered))[:, None]
    sample_loose_specific_model = LedoitWolf(assume_centered=True)
    sample_loose_specific_model.fit(sample_loose_residual_weighted)
    sample_loose_specific_covariance = pd.DataFrame(
        sample_loose_specific_model.covariance_ * 252 / factor_horizon,
        index=assets,
        columns=assets,
    )

    sample_tight_residual_mean = sample_residuals.mul(sample_tight_weight, axis=0).sum()
    sample_tight_residual_centered = sample_residuals - sample_tight_residual_mean
    sample_tight_residual_weighted = sample_tight_residual_centered.values * np.sqrt(sample_tight_weight.values * len(sample_tight_residual_centered))[:, None]
    sample_tight_specific_model = LedoitWolf(assume_centered=True)
    sample_tight_specific_model.fit(sample_tight_residual_weighted)
    sample_tight_specific_covariance = pd.DataFrame(
        sample_tight_specific_model.covariance_ * 252 / factor_horizon,
        index=assets,
        columns=assets,
    )

    sample_specific_covariance = (
        sample_loose_specific_covariance * sample_current_probability["Loose"]
        + sample_tight_specific_covariance * sample_current_probability["Tight"]
    )
    sample_specific_covariance = pd.DataFrame(
        np.diag(np.diag(sample_specific_covariance.values)),
        index=assets,
        columns=assets,
    )

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
    current_linear_factor_variance = float(current_factor_exposure.values @ sample_factor_covariance.values @ current_factor_exposure.values.T)
    current_factor_variance = current_linear_factor_variance
    current_specific_variance = float(current_weight.values @ sample_specific_covariance.values @ current_weight.values)
    current_total_variance = current_factor_variance + current_specific_variance

    factor_contribution = current_factor_exposure.values * (sample_factor_covariance.values @ current_factor_exposure.values)
    factor_row = pd.Series(factor_contribution / current_total_variance, index=sample_factors.columns, name=current_date)
    rolling_factor_contribution.append(factor_row)

    split_row = pd.Series(
        {
            "Factor Risk": current_factor_variance / current_total_variance,
            "Specific Risk": current_specific_variance / current_total_variance,
            "Predicted Volatility": np.sqrt(current_total_variance),
        },
        name=current_date,
    )
    rolling_factor_specific.append(split_row)

    realized_sample = daily_asset_returns[(daily_asset_returns.index > current_date) & (daily_asset_returns.index <= current_date + pd.DateOffset(months=12))]
    if len(realized_sample) >= 126:
        realized_portfolio_returns = realized_sample[assets] @ current_weight
        realized_volatility = realized_portfolio_returns.std() * np.sqrt(252)
        validation_rows.append(
            {
                "forecast_date": current_date,
                "realized_start_date": realized_sample.index[0],
                "realized_end_date": realized_sample.index[-1],
                "predicted_volatility": np.sqrt(current_total_variance),
                "realized_volatility": realized_volatility,
                "forecast_error": realized_volatility - np.sqrt(current_total_variance),
            }
        )

rolling_factor_contribution = pd.DataFrame(rolling_factor_contribution)
rolling_factor_specific = pd.DataFrame(rolling_factor_specific)

if len(rolling_factor_contribution) > 0:
    rolling_factor_group = pd.DataFrame(index=rolling_factor_contribution.index)
    rolling_factor_group["Regime Tightness Risk"] = rolling_factor_contribution[regime_factor_names].sum(axis=1)
    rolling_factor_group["Equity Market Beta"] = rolling_factor_contribution["Equity Market Beta"]
    rolling_factor_group["TIPS Duration Beta"] = rolling_factor_contribution["TIPS Duration Beta"]
    rolling_factor_group["Credit Market Beta"] = rolling_factor_contribution["Credit Market Beta"]
    rolling_factor_group["Downside Market Shock"] = rolling_factor_contribution["Downside Market Shock"]
    rolling_factor_group["Real Yield Shock"] = rolling_factor_contribution["Real Yield Shock"]
    rolling_factor_group["Specific Risk"] = rolling_factor_specific["Specific Risk"]
else:
    rolling_factor_group = pd.DataFrame()

if len(validation_rows) > 0:
    validation_table = pd.DataFrame(validation_rows).set_index("forecast_date")
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
loose_factor_covariance.to_csv(output_folder + "/loose_factor_covariance.csv")
tight_factor_covariance.to_csv(output_folder + "/tight_factor_covariance.csv")
factor_covariance.to_csv(output_folder + "/factor_covariance.csv")
factor_correlation.to_csv(output_folder + "/factor_correlation_matrix.csv")
loose_specific_covariance.to_csv(output_folder + "/loose_specific_covariance.csv")
tight_specific_covariance.to_csv(output_folder + "/tight_specific_covariance.csv")
specific_covariance.to_csv(output_folder + "/specific_covariance_shrunk.csv")
factor_model_covariance.to_csv(output_folder + "/factor_model_covariance.csv")
sample_covariance.to_csv(output_folder + "/sample_covariance.csv")
portfolio_factor_exposure.to_csv(output_folder + "/portfolio_factor_exposure.csv")
factor_risk_contribution.to_csv(output_folder + "/factor_risk_contribution.csv")
factor_group_contribution.to_csv(output_folder + "/factor_group_risk_contribution.csv")
specific_risk_contribution.to_csv(output_folder + "/specific_risk_contribution.csv")
asset_risk_contribution.to_csv(output_folder + "/asset_risk_contribution.csv")
portfolio_summary.to_csv(output_folder + "/portfolio_risk_summary.csv")
factor_specific_split.to_csv(output_folder + "/factor_vs_specific_risk.csv")
exposure_interpretation.to_csv(output_folder + "/asset_factor_exposure_interpretation.csv", index=False)
stress_summary.to_csv(output_folder + "/stress_scenario_summary.csv")
stress_table.to_csv(output_folder + "/stress_scenario_by_asset.csv")
rolling_factor_contribution.to_csv(output_folder + "/rolling_factor_risk_contribution.csv")
rolling_factor_group.to_csv(output_folder + "/rolling_factor_group_risk_contribution.csv")
rolling_factor_specific.to_csv(output_folder + "/rolling_factor_specific_risk.csv")
validation_table.to_csv(output_folder + "/predicted_vs_realized_volatility.csv")
validation_metrics.to_csv(output_folder + "/validation_metrics.csv")

beta.round(3).to_csv(table_folder + "/factor_exposures.csv")
factor_risk_contribution.round(4).to_csv(table_folder + "/factor_risk_contribution.csv")
factor_group_contribution.round(4).to_csv(table_folder + "/factor_group_risk_contribution.csv")
asset_risk_contribution.round(4).to_csv(table_folder + "/asset_risk_contribution.csv")
portfolio_summary.round(4).to_csv(table_folder + "/portfolio_risk_summary.csv")
factor_specific_split.round(4).to_csv(table_folder + "/factor_vs_specific_risk.csv")
factor_correlation.round(3).to_csv(table_folder + "/factor_correlation_matrix.csv")
exposure_interpretation.round(4).to_csv(table_folder + "/asset_factor_exposure_interpretation.csv", index=False)
stress_summary.round(4).to_csv(table_folder + "/stress_scenario_summary.csv")
stress_table.round(4).to_csv(table_folder + "/stress_scenario_by_asset.csv")
rolling_factor_contribution.round(4).to_csv(table_folder + "/rolling_factor_risk_contribution.csv")
rolling_factor_group.round(4).to_csv(table_folder + "/rolling_factor_group_risk_contribution.csv")
rolling_factor_specific.round(4).to_csv(table_folder + "/rolling_factor_specific_risk.csv")
validation_table.round(4).to_csv(table_folder + "/predicted_vs_realized_volatility.csv")
validation_metrics.round(4).to_csv(table_folder + "/validation_metrics.csv")

print("\nPortfolio risk summary:")
print(portfolio_summary.round(4))

print("\nLargest factor risk contributions:")
print(factor_risk_contribution.sort_values(ascending=False).round(4))

print("\nFactor group risk contribution:")
print(factor_group_contribution.round(4))

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


plt.figure(figsize=(10, 8))
plt.imshow(factor_correlation.values, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(np.arange(len(factor_correlation.columns)), factor_correlation.columns, rotation=45, ha="right")
plt.yticks(np.arange(len(factor_correlation.index)), factor_correlation.index)
plt.title("Factor Correlation Matrix", fontsize=22, fontweight="bold")
plt.tight_layout()
plt.savefig(figure_folder + "/Factor Correlation Matrix.png", dpi=200)


plt.figure(figsize=(12, 6))
plot_data = factor_group_contribution.drop("Specific Risk").sort_values(ascending=False)
plt.bar(plot_data.index, plot_data.values, color="#1f77b4")
plt.axhline(0, color="black", linewidth=1)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.title("Portfolio Factor Risk Contribution", fontsize=22, fontweight="bold")
plt.ylabel("Share of portfolio variance")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(figure_folder + "/Portfolio Factor Risk Contribution.png", dpi=200)


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


if len(rolling_factor_contribution) > 0:
    plt.figure(figsize=(14, 7))
    rolling_plot = rolling_factor_group.drop(columns=["Specific Risk"], errors="ignore").rolling(3).mean()
    positive_plot = rolling_plot.clip(lower=0)
    negative_plot = rolling_plot.clip(upper=0)
    colors = ["#1f77b4", "#8c564b", "#bcbd22", "#2ca02c", "#ff7f0e", "#d62728"]
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


if len(validation_table) > 0:
    plt.figure(figsize=(13, 6))
    plt.plot(validation_table.index, validation_table["predicted_volatility"], linewidth=2.5, label="Model-Implied Volatility")
    plt.plot(validation_table["realized_end_date"], validation_table["realized_volatility"], linewidth=2.5, label="Forward Realized Volatility")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.title("Predicted vs Realized Volatility", fontsize=22, fontweight="bold")
    plt.ylabel("Annualized volatility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_folder + "/Predicted vs Realized Volatility.png", dpi=200)


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
