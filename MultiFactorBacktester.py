"""
Multi-factor backtester for US equities using momentum and low volatility factors.

This module provides two primary classes:
    - DataHandler: Downloads and prepares price data and factor signals.
    - Backtester: Executes a monthly rebalanced, equal-weight portfolio strategy.

The implementation emphasizes avoiding look-ahead bias by only using data
available up to each rebalance date when forming positions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# Configure logging for transparency during data processing and backtests.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ANNUAL_RISK_FREE_RATE: float = 0.03
TRADING_DAYS_PER_YEAR: int = 252


@dataclass
class DataHandler:
    """Load, clean, and compute factors for a set of equities.

    Attributes:
        tickers: List of ticker symbols to download.
        start_date: Inclusive start date for historical data.
        end_date: Inclusive end date for historical data.
        prices: Adjusted close prices indexed by date.
        returns: Daily simple returns derived from adjusted close prices.
    """

    tickers: List[str]
    start_date: str
    end_date: str
    prices: pd.DataFrame | None = None
    returns: pd.DataFrame | None = None

    def download_data(self) -> pd.DataFrame:
        """Download adjusted close prices and compute daily returns.

        Returns:
            DataFrame of adjusted close prices indexed by date and ticker columns.
        """
        logging.info("Downloading data for tickers: %s", ", ".join(self.tickers))
        data = yf.download(
            tickers=self.tickers,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=False,
            progress=False,
        )
        # yfinance returns a multi-index column; select the adjusted close slice.
        adj_close = data["Adj Close"].dropna(how="all")
        adj_close = adj_close.sort_index()

        # Forward-fill within each column to handle small gaps.
        adj_close = adj_close.ffill()

        self.prices = adj_close
        self.returns = adj_close.pct_change().fillna(0.0)

        logging.info("Downloaded %d rows of price data", len(adj_close))
        return adj_close

    def calculate_factors(self) -> pd.DataFrame:
        """Compute momentum and low volatility factors using historical data.

        Momentum: 12-month simple return (price / price from 252 trading days ago - 1).
        Low Volatility: Annualized volatility of the past 252 trading days of returns,
        scaled by sqrt(252) to express as annualized standard deviation.
        Returns: Daily simple returns to drive the PnL calculation.

        Returns:
            DataFrame with a MultiIndex on columns: (factor, ticker) for
            factors ['momentum', 'low_vol', 'returns'], aligned by date.
        """
        if self.prices is None or self.returns is None:
            raise ValueError("Call download_data() before calculating factors.")

        lookback = TRADING_DAYS_PER_YEAR

        # Momentum uses past price relatives; shift to avoid look-ahead when selecting.
        momentum = self.prices.pct_change(periods=lookback).shift(1)

        # Rolling volatility of daily returns, annualized; shifted to avoid look-ahead.
        rolling_vol = (
            self.returns.rolling(window=lookback, min_periods=lookback)
            .std()
            .multiply(np.sqrt(TRADING_DAYS_PER_YEAR))
            .shift(1)
        )

        factor_frames = {
            "momentum": momentum,
            "low_vol": rolling_vol,
            "returns": self.returns,
        }
        factor_data = pd.concat(
            factor_frames, axis=1
        )  # Columns become MultiIndex (factor, ticker)

        logging.info("Calculated factors with shape %s", factor_data.shape)
        return factor_data


class Backtester:
    """Execute a factor-based equity strategy with monthly rebalancing."""

    def __init__(self, factor_data: pd.DataFrame, initial_capital: float = 100_000):
        """Initialize the backtester with factor and return data.

        Args:
            factor_data: DataFrame with MultiIndex columns (factor, ticker) including
                'momentum' and 'low_vol'. The index must be datetime.
            initial_capital: Starting portfolio value.
        """
        required_factors = {"momentum", "low_vol", "returns"}
        provided_factors = set(factor_data.columns.get_level_values(0))
        missing = required_factors - provided_factors
        if missing:
            raise ValueError(f"Factor data missing required factors: {missing}")

        self.factor_data = factor_data.sort_index()
        self.initial_capital = initial_capital

    def run_backtest(
        self, lookback_periods: int, holding_period: int, n_stocks: int
    ) -> pd.DataFrame:
        """Run the monthly-rebalanced strategy.

        Args:
            lookback_periods: Number of trading days used for factor computation
                (kept for transparency; factors already computed in DataHandler).
            holding_period: Holding horizon in trading days after each rebalance.
            n_stocks: Number of stocks to hold at each rebalance.

        Returns:
            DataFrame with daily portfolio and benchmark equity curves.
        """
        momentum = self.factor_data.xs("momentum", level=0, axis=1)
        low_vol = self.factor_data.xs("low_vol", level=0, axis=1)
        returns = self.factor_data.xs("returns", level=0, axis=1)

        # First valid date where both factors exist.
        valid_mask = momentum.notna() & low_vol.notna()
        valid_dates = valid_mask.any(axis=1)
        if not valid_dates.any():
            raise ValueError("No valid factor data available to run the backtest.")
        first_valid_date = valid_dates[valid_dates].index[0]

        # Rebalance on the first trading day of each month.
        rebalance_dates = (
            returns.loc[first_valid_date:].resample("MS").first().index.to_list()
        )

        portfolio_daily_returns = pd.Series(0.0, index=returns.index)
        benchmark_returns = returns.mean(axis=1)

        for i, rebalance_date in enumerate(rebalance_dates):
            if rebalance_date not in returns.index:
                # Skip if the first calendar day is not a trading day in the dataset.
                continue
            start_idx = returns.index.get_loc(rebalance_date)
            next_rebalance_date = (
                rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else None
            )
            end_idx = len(returns) - 1
            if next_rebalance_date is not None and next_rebalance_date in returns.index:
                end_idx = min(end_idx, returns.index.get_loc(next_rebalance_date) - 1)
            end_idx = min(end_idx, start_idx + holding_period - 1)
            if end_idx < start_idx:
                continue

            # Select factors available on rebalance_date, drop assets with missing signals.
            factor_snapshot = pd.DataFrame(
                {
                    "momentum": momentum.iloc[start_idx],
                    "low_vol": low_vol.iloc[start_idx],
                }
            ).dropna()
            if factor_snapshot.empty:
                continue

            low_vol_cutoff = factor_snapshot["low_vol"].quantile(0.25)
            candidate_universe = factor_snapshot[
                factor_snapshot["low_vol"] <= low_vol_cutoff
            ]
            selected = (
                candidate_universe.sort_values("momentum", ascending=False)
                .head(n_stocks)
                .index
            )

            if len(selected) == 0:
                slice_index = returns.index[start_idx : end_idx + 1]
                portfolio_daily_returns.loc[slice_index] = 0.0
                continue

            weights = pd.Series(1.0 / len(selected), index=selected)
            period_returns = returns.loc[
                returns.index[start_idx : end_idx + 1], selected
            ]
            weighted_returns = period_returns.mul(weights, axis=1).sum(axis=1)
            portfolio_daily_returns.loc[weighted_returns.index] = weighted_returns

        portfolio_equity = (
            (1 + portfolio_daily_returns).cumprod() * self.initial_capital
        )
        benchmark_equity = (1 + benchmark_returns).cumprod() * self.initial_capital

        result = pd.DataFrame(
            {
                "portfolio": portfolio_equity,
                "benchmark": benchmark_equity,
                "portfolio_returns": portfolio_daily_returns,
                "benchmark_returns": benchmark_returns,
            }
        )
        return result

    def calculate_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Compute key performance statistics.

        Args:
            returns: Daily portfolio returns.

        Returns:
            Dictionary with total_return, annual_volatility, max_drawdown,
            and annual_sharpe_ratio.
        """
        rf_daily = (1 + ANNUAL_RISK_FREE_RATE) ** (1 / TRADING_DAYS_PER_YEAR) - 1
        mean_return = returns.mean()
        vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)  # annualized sigma
        total_return = (1 + returns).prod() - 1
        sharpe = 0.0
        if vol > 0:
            sharpe = ((mean_return - rf_daily) / vol) * np.sqrt(TRADING_DAYS_PER_YEAR)

        equity_curve = (1 + returns).cumprod()
        rolling_max = equity_curve.cummax()
        drawdowns = equity_curve / rolling_max - 1
        max_drawdown = drawdowns.min()

        return {
            "total_return": float(total_return),
            "annual_volatility": float(vol),
            "max_drawdown": float(max_drawdown),
            "annual_sharpe_ratio": float(sharpe),
        }

    def plot_results(self, equity_curve: pd.DataFrame) -> None:
        """Plot portfolio vs benchmark equity curves.

        Args:
            equity_curve: DataFrame with 'portfolio' and 'benchmark' columns.
        """
        plt.figure(figsize=(10, 6))
        equity_curve[["portfolio", "benchmark"]].plot(ax=plt.gca())
        plt.title("Portfolio vs Benchmark Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Equity ($)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Example execution with Dow 30 constituents.
    dow_30 = [
        "AAPL",
        "AMGN",
        "AXP",
        "BA",
        "CAT",
        "CRM",
        "CSCO",
        "CVX",
        "DIS",
        "DOW",
        "GS",
        "HD",
        "HON",
        "IBM",
        "INTC",
        "JNJ",
        "JPM",
        "KO",
        "MCD",
        "MMM",
        "MRK",
        "MSFT",
        "NKE",
        "PG",
        "TRV",
        "UNH",
        "V",
        "VZ",
        "WBA",
        "WMT",
    ]

    start = "2010-01-01"
    end = pd.Timestamp.today().strftime("%Y-%m-%d")

    data_handler = DataHandler(dow_30, start_date=start, end_date=end)
    data_handler.download_data()
    factors = data_handler.calculate_factors()

    backtester = Backtester(factors, initial_capital=100_000)
    equity = backtester.run_backtest(
        lookback_periods=TRADING_DAYS_PER_YEAR, holding_period=21, n_stocks=5
    )
    metrics = backtester.calculate_metrics(equity["portfolio_returns"])

    print("Performance metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    backtester.plot_results(equity[["portfolio", "benchmark"]])

