# MultiFactor Backtester

Object-oriented Python backtester for a two-factor, long-only equity strategy (12‑month momentum and low volatility) with monthly rebalancing, equal weights, and production-minded safeguards against look-ahead bias.

## Features
- Downloads and cleans Adjusted Close data via `yfinance`.
- Computes 12-month momentum and annualized 1-year volatility (shifted to prevent look-ahead).
- Monthly rebalance: select lowest-volatility quartile then pick top‑momentum names; equal-weighted.
- Tracks portfolio vs. equal-weight benchmark, daily returns, equity curves, and performance stats (total return, annualized vol, max drawdown, annualized Sharpe using a 3% risk-free rate).
- Plotting utility for equity curves and a runnable example using the Dow 30.

## Project Structure
- `MultiFactorBacktester.py`: data handling, factor computation, backtest engine, metrics, plotting, and example run block.

## Setup
```bash
cd /Users/bu/IdeaProjects/Factor-Model-Simulator
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy yfinance matplotlib
```

## How It Works
1) **DataHandler**
   - `download_data()`: pulls Adjusted Close prices, forward-fills gaps, computes daily simple returns.
   - `calculate_factors()`: builds a MultiIndex DataFrame with:
     - `momentum`: 12‑month return, shifted one day.
     - `low_vol`: 1‑year rolling volatility, annualized via `sqrt(252)`, shifted one day.
     - `returns`: daily simple returns.
2) **Backtester**
   - Rebalance on first trading day of each month.
   - Filter tickers to lowest 25% by `low_vol`, then pick top `n_stocks` by `momentum`; equal weights.
   - Hold for `holding_period` trading days; compute daily portfolio and benchmark returns/equity.
   - `calculate_metrics()`: total return, annualized volatility, max drawdown, annualized Sharpe (3% rf).
   - `plot_results()`: equity curve vs. benchmark.

Look-ahead bias is avoided by shifting factor inputs one day and only using data available as of each rebalance date.

## Running the Example
From the project root after activating the venv:
```bash
python MultiFactorBacktester.py
```
Defaults:
- Universe: Dow 30 tickers.
- Dates: `2010-01-01` to today.
- Params: `lookback_periods=252`, `holding_period=21`, `n_stocks=5`, `initial_capital=100_000`.

## Customization
- Change universe: edit the `dow_30` list in `__main__`.
- Adjust dates: set `start`/`end` in `__main__`.
- Tune strategy: modify `lookback_periods`, `holding_period`, `n_stocks`, or `initial_capital` in the example run.

## Notes
- Network access is required for `yfinance` downloads.
- Equity curve plot will display interactively; ensure a display backend is available (or switch matplotlib backend as needed).