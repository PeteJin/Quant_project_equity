# Regime-Aware Equity Portfolio

This project builds a regime-aware quantitative strategy pipeline for a concentrated mock equity portfolio. The portfolio covers defensive consumer names, industrials, financial exposure, commodities, and duration hedges. The pipeline converts macro regime signals into forward return views, links those views to Black-Litterman posterior estimates, and allocates capital through multiple optimization methods with a focus on risk-adjusted performance and controlled downside exposure.

The workflow uses daily macro inputs to estimate tight and loose market regimes with a Hidden Markov Model. Regime probabilities are translated into forward return views, mapped into Black-Litterman posterior returns and covariance, and then passed into several optimizers: Black-Litterman max Sharpe, minimum variance, semivariance, CVaR, HRP, and NCO. The final section runs a rolling monthly backtest and a one-year correlated GBM Monte Carlo simulation using Cholesky-decomposed covariance shocks.

## Project Structure

```text
data/raw/macro/                         raw macro inputs
data/processed/project1/                generated model and backtest tables
reports/figures/project1/               final charts
reports/tables/project1/                final summary tables
src/equity_quant_project/project1/      project code
```

Main files:

```text
hmm.py          daily macro cleaning and HMM regime detection
views.py        regime-conditioned return views
bl.py           Black-Litterman posterior returns, covariance, and allocation
optimizers.py   MVO, semivariance, CVaR, HRP, and NCO allocation analysis
backtest.py     rolling monthly out-of-sample backtest
simulation.py   Cholesky GBM Monte Carlo simulation
```

## Run Order

Run the files from the repository root in this order:

```text
1. src/equity_quant_project/project1/hmm.py
2. src/equity_quant_project/project1/views.py
3. src/equity_quant_project/project1/bl.py
4. src/equity_quant_project/project1/optimizers.py
5. src/equity_quant_project/project1/backtest.py
6. src/equity_quant_project/project1/simulation.py
```

The files can also be imported as package modules:

```python
from equity_quant_project.project1 import hmm
from equity_quant_project.project1 import views
from equity_quant_project.project1 import bl
```

## Environment

```bash
pip install -r requirements.txt
pip install -e .
```

## Outputs

Key outputs include:

```text
Regime Posterior Probabilities.png
Black-Litterman Return Views.png
Black-Litterman Posterior Covariance Matrix.png
Optimizer Allocations.png
Rolling Regime-Aware Portfolio Backtest.png
Black-Litterman vs Benchmarks.png
Monte-Carlo Terminal Return Distributions.png
```

Processed model tables and backtest results are written to:

```text
data/processed/project1/
```

## Backtest Summary

<img src="reports/figures/project1/Rolling%20Regime-Aware%20Portfolio%20Backtest.png" alt="Rolling Regime-Aware Portfolio Backtest" width="900">

| Strategy | Annual Return | Annual Volatility | Excess vs Equal Weight | Excess vs S&P 500 | Sharpe | Sortino | Max Drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| BL | 20.81% | 15.68% | 4.62% | 6.27% | 1.07 | 2.12 | -19.31% |
| MVO Min Vol | 19.86% | 14.92% | 3.67% | 5.32% | 1.06 | 2.11 | -18.00% |
| HRP | 19.33% | 14.67% | 3.14% | 4.79% | 1.05 | 2.09 | -17.80% |
| NCO | 19.71% | 14.85% | 3.52% | 5.17% | 1.06 | 2.12 | -17.93% |
| Equal Weight | 16.19% | 11.52% | 0.00% | 1.65% | 1.06 | 1.68 | -16.18% |
| S&P 500 | 14.54% | 14.42% | -1.65% | 0.00% | 0.73 | 1.10 | -23.93% |
