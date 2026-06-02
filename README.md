# Regime-Aware Equity Portfolio

This project builds a regime-aware quantitative strategy pipeline for a concentrated mock equity portfolio. The portfolio covers defensive consumer names, industrials, financial exposure, commodities, and duration hedges. The pipeline converts macro regime signals into forward return views, links those views to Black-Litterman posterior estimates, and allocates capital through multiple optimization methods with a focus on risk-adjusted performance and controlled downside exposure.

The workflow uses daily macro inputs to estimate tight and loose market regimes with a Hidden Markov Model. Regime probabilities are translated into forward return views, mapped into Black-Litterman posterior returns and covariance, and then passed into several optimizers: Black-Litterman max Sharpe, minimum variance, semivariance, CVaR, HRP, and NCO. The final section runs a rolling monthly backtest and a one-year correlated GBM Monte Carlo simulation using Cholesky-decomposed covariance shocks.

## Project Structure

```text
data/raw/macro/                         raw macro inputs
data/processed/project1/                generated model and backtest tables
data/processed/project3/                generated risk model tables
reports/figures/project1/               final charts
reports/figures/project3/               factor risk charts
reports/tables/project1/                final summary tables
reports/tables/project3/                factor risk summary tables
src/equity_quant_project/project1/      project code
src/equity_quant_project/project3/      portfolio risk model code
```

Main files:

Project 1:

```text
hmm.py          daily macro cleaning and HMM regime detection
views.py        regime-conditioned return views
bl.py           Black-Litterman posterior returns, covariance, and allocation
optimizers.py   MVO, semivariance, CVaR, HRP, and NCO allocation analysis
backtest.py     rolling monthly out-of-sample backtest
simulation.py   Cholesky GBM Monte Carlo simulation
```

Project 3:

```text
risk_factor_model.py   portfolio factor risk model
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
7. src/equity_quant_project/project3/risk_factor_model.py
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
Portfolio Factor Risk Contribution.png
Rolling Factor Risk Attribution.png
Predicted vs Realized Volatility.png
```

Processed model tables and backtest results are written to:

```text
data/processed/project1/
data/processed/project3/
```

## Backtest Summary

<img src="reports/figures/project1/Rolling%20Regime-Aware%20Portfolio%20Backtest.png" alt="Rolling Regime-Aware Portfolio Backtest" width="900">

| Strategy | Annual Return | Annual Volatility | Excess vs Equal Weight | Excess vs S&P 500 | Sharpe | Sortino | Recovery |
|---|---:|---:|---:|---:|---:|---:|---:|
| BL | 20.81% | 15.68% | 4.62% | 6.27% | 1.07 | 2.12 | 5 |
| MVO Min Vol | 19.86% | 14.92% | 3.67% | 5.32% | 1.06 | 2.11 | 4 |
| HRP | 19.33% | 14.67% | 3.14% | 4.79% | 1.05 | 2.09 | 5 |
| NCO | 19.71% | 14.85% | 3.52% | 5.17% | 1.06 | 2.12 | 4 |
| Equal Weight | 16.19% | 11.52% | 0.00% | 1.65% | 1.06 | 1.68 | 4 |
| S&P 500 | 14.54% | 14.42% | -1.65% | 0.00% | 0.73 | 1.10 | 15 |

## Selected Figures

### Regime Posterior Probabilities

![Regime Posterior Probabilities](reports/figures/project1/Regime%20Posterior%20Probabilities.png)

### Regime-Conditioned Stock Returns

![Regime-Conditioned Stock Returns](reports/figures/project1/Regime-Conditioned%20Stock%20Returns.png)

### Black-Litterman Return Views

![Black-Litterman Return Views](reports/figures/project1/Black-Litterman%20Return%20Views.png)

### Black-Litterman Posterior Covariance Matrix

![Black-Litterman Posterior Covariance Matrix](reports/figures/project1/Black-Litterman%20Posterior%20Covariance%20Matrix.png)

### Efficient Frontier

![Efficient Frontier BL](reports/figures/project1/Efficient%20Frontier%20BL.png)

### NCO Cluster Correlation Map

![NCO Cluster Correlation Map](reports/figures/project1/NCO%20Cluster%20Correlation%20Map.png)

### Optimizer Allocations

![Optimizer Allocations](reports/figures/project1/Optimizer%20Allocations.png)

## Project 3 - Portfolio Factor Risk Model

Project 3 extends the portfolio work into a Barra-style risk attribution analysis of the same mock portfolio universe. The model decomposes portfolio risk into systematic factor risk and stock-specific risk, then tracks the change of risk contributions through market conditions.

The model uses the HMM regime inputs as the first layer of risk factors, then adds institutional market factors and custom macro shocks:

- **Equity Market Beta**: uses SPY as the broad equity market factor
- **TIPS Duration Beta**: uses TIP as the inflation-linked duration factor
- **Credit Market Beta**: uses HYG as the credit beta factor
- **Downside Market Shock**: measures sensitivity to downside equity market moves
- **Real Yield Shock**: uses real-rate pressure as a macro discount-rate risk channel

The workflow aligns factor observations with forward 21-trading-day asset returns, purges the market beta factors against the regime inputs, estimates asset-level exposures with EWMA-weighted WLS, builds Ledoit-Wolf covariance estimates, separates factor and specific risk, and applies Euler risk decomposition to attribute portfolio variance across factors and individual assets. It also includes rolling risk attribution, model-implied versus forward realized volatility validation, factor correlation analysis, and a stress scenario based on tightening macro conditions, rising real yields, and credit pressure.

Key outputs include:

```text
Factor exposure by asset
Portfolio factor exposure
Factor risk contribution
Asset-level risk contribution
Factor risk vs specific risk split
Rolling factor risk attribution
Predicted vs realized volatility validation
Sample vs factor model covariance comparison
Macro stress scenario impact
```

### Barra-Style Factor Exposures

![Barra-Style Factor Exposures](reports/figures/project3/Barra-Style%20Factor%20Exposures.png)

### Portfolio Factor Risk Contribution

![Portfolio Factor Risk Contribution](reports/figures/project3/Portfolio%20Factor%20Risk%20Contribution.png)

### Rolling Factor Risk Attribution

![Rolling Factor Risk Attribution](reports/figures/project3/Rolling%20Factor%20Risk%20Attribution.png)

### Predicted vs Forward Realized Volatility

![Predicted vs Forward Realized Volatility](reports/figures/project3/Predicted%20vs%20Realized%20Volatility.png)

### Sample vs Factor Model Covariance

![Sample vs Factor Model Covariance](reports/figures/project3/Sample%20vs%20Factor%20Model%20Covariance.png)

### Stress Scenario by Asset

![Stress Scenario by Asset](reports/figures/project3/Stress%20Scenario%20by%20Asset.png)
