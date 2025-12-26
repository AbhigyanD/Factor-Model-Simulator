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

    def run_backtest_enhanced(
        self,
        lookback_periods: int,
        holding_period: int,
        n_stocks: int,
        strategy: str = "composite_score",
        vol_percentile: float = 0.5,
        momentum_weight: float = 0.7,
        use_momentum_weighting: bool = False,
    ) -> pd.DataFrame:
        """Run enhanced backtest with multiple strategy options.

        Strategy options:
        - "composite_score": Combines momentum and low_vol into a single score
        - "risk_adjusted_momentum": Uses momentum/volatility ratio (Sharpe-like)
        - "relaxed_vol_filter": Filters by volatility percentile, then momentum
        - "momentum_first": Filters by momentum first, then low volatility
        - "momentum_only": Pure momentum strategy (no volatility filter)

        Args:
            lookback_periods: Number of trading days used for factor computation.
            holding_period: Holding horizon in trading days after each rebalance.
            n_stocks: Number of stocks to hold at each rebalance.
            strategy: Strategy selection method (see options above).
            vol_percentile: Volatility percentile threshold (0.25 = bottom 25%, 0.5 = bottom 50%).
            momentum_weight: Weight for momentum in composite score (0-1).
            use_momentum_weighting: If True, weight positions by momentum strength.

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

            # Get factor snapshot
            factor_snapshot = pd.DataFrame(
                {
                    "momentum": momentum.iloc[start_idx],
                    "low_vol": low_vol.iloc[start_idx],
                }
            ).dropna()
            if factor_snapshot.empty:
                continue

            # Strategy selection logic
            if strategy == "composite_score":
                # Normalize factors to z-scores for fair combination
                mom_z = (factor_snapshot["momentum"] - factor_snapshot["momentum"].mean()) / (
                    factor_snapshot["momentum"].std() + 1e-8
                )
                vol_z = (
                    factor_snapshot["low_vol"].mean() - factor_snapshot["low_vol"]
                ) / (factor_snapshot["low_vol"].std() + 1e-8)  # Inverted (lower vol = better)
                composite = momentum_weight * mom_z + (1 - momentum_weight) * vol_z
                selected = composite.nlargest(n_stocks).index

            elif strategy == "risk_adjusted_momentum":
                # Momentum divided by volatility (Sharpe-like ratio)
                risk_adj_mom = factor_snapshot["momentum"] / (factor_snapshot["low_vol"] + 1e-8)
                selected = risk_adj_mom.nlargest(n_stocks).index

            elif strategy == "relaxed_vol_filter":
                # Filter by volatility percentile, then momentum
                vol_cutoff = factor_snapshot["low_vol"].quantile(vol_percentile)
                candidate_universe = factor_snapshot[
                    factor_snapshot["low_vol"] <= vol_cutoff
                ]
                if len(candidate_universe) == 0:
                    selected = pd.Index([])
                else:
                    selected = (
                        candidate_universe.sort_values("momentum", ascending=False)
                        .head(n_stocks)
                        .index
                    )

            elif strategy == "momentum_first":
                # Filter by momentum first, then low volatility
                top_momentum = factor_snapshot.nlargest(n_stocks * 2, "momentum")
                selected = top_momentum.nsmallest(n_stocks, "low_vol").index

            elif strategy == "momentum_only":
                # Pure momentum strategy
                selected = factor_snapshot.nlargest(n_stocks, "momentum").index

            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            if len(selected) == 0:
                slice_index = returns.index[start_idx : end_idx + 1]
                portfolio_daily_returns.loc[slice_index] = 0.0
                continue

            # Weighting scheme
            if use_momentum_weighting and len(selected) > 0:
                # Weight by momentum strength (normalized)
                selected_momentum = factor_snapshot.loc[selected, "momentum"]
                weights = selected_momentum / selected_momentum.sum()
            else:
                # Equal weighting
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

    def plot_results(self, equity_curve: pd.DataFrame, title: str = "Portfolio vs Benchmark Equity Curve") -> None:
        """Plot portfolio vs benchmark equity curves.

        Args:
            equity_curve: DataFrame with 'portfolio' and 'benchmark' columns.
            title: Plot title.
        """
        plt.figure(figsize=(10, 6))
        equity_curve[["portfolio", "benchmark"]].plot(ax=plt.gca())
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Equity ($)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def compare_strategies(
        self,
        lookback_periods: int,
        holding_period: int,
        n_stocks: int,
        strategies: List[str] | None = None,
    ) -> pd.DataFrame:
        """Compare multiple strategies side-by-side.

        Args:
            lookback_periods: Number of trading days used for factor computation.
            holding_period: Holding horizon in trading days after each rebalance.
            n_stocks: Number of stocks to hold at each rebalance.
            strategies: List of strategy names to test. If None, tests all strategies.

        Returns:
            DataFrame with metrics for each strategy.
        """
        if strategies is None:
            strategies = [
                "composite_score",
                "risk_adjusted_momentum",
                "relaxed_vol_filter",
                "momentum_first",
                "momentum_only",
            ]

        results = []
        for strategy in strategies:
            try:
                equity = self.run_backtest_enhanced(
                    lookback_periods=lookback_periods,
                    holding_period=holding_period,
                    n_stocks=n_stocks,
                    strategy=strategy,
                )
                metrics = self.calculate_metrics(equity["portfolio_returns"])
                metrics["strategy"] = strategy
                results.append(metrics)
            except Exception as e:
                logging.warning(f"Strategy {strategy} failed: {e}")

        return pd.DataFrame(results)

    def monte_carlo_simulation(
        self,
        base_lookback: int = 252,
        base_holding_period: int = 21,
        base_n_stocks: int = 5,
        strategy: str = "risk_adjusted_momentum",
        n_simulations: int = 100,
        param_ranges: Dict[str, tuple] | None = None,
        random_seed: int | None = None,
    ) -> pd.DataFrame:
        """Run Monte Carlo simulation with randomized parameters.

        Tests strategy robustness by varying parameters within specified ranges
        and collecting performance statistics across all runs.

        Args:
            base_lookback: Base lookback period (will be varied).
            base_holding_period: Base holding period in days (will be varied).
            base_n_stocks: Base number of stocks (will be varied).
            strategy: Strategy to test.
            n_simulations: Number of Monte Carlo runs.
            param_ranges: Dict with parameter ranges, e.g.:
                {
                    'n_stocks': (3, 8),  # Range for n_stocks
                    'holding_period': (15, 30),  # Range for holding_period
                    'momentum_weight': (0.5, 0.9),  # For composite_score
                }
            random_seed: Random seed for reproducibility.

        Returns:
            DataFrame with metrics for each simulation run.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        if param_ranges is None:
            param_ranges = {
                "n_stocks": (3, 8),
                "holding_period": (15, 30),
                "momentum_weight": (0.5, 0.9),
            }

        results = []
        logging.info(f"Running {n_simulations} Monte Carlo simulations...")

        for i in range(n_simulations):
            # Randomize parameters
            n_stocks = int(
                np.random.uniform(param_ranges["n_stocks"][0], param_ranges["n_stocks"][1])
            )
            holding_period = int(
                np.random.uniform(
                    param_ranges["holding_period"][0], param_ranges["holding_period"][1]
                )
            )

            # Strategy-specific parameters
            momentum_weight = None
            if strategy == "composite_score":
                momentum_weight = np.random.uniform(
                    param_ranges.get("momentum_weight", (0.5, 0.9))[0],
                    param_ranges.get("momentum_weight", (0.5, 0.9))[1],
                )

            vol_percentile = None
            if strategy == "relaxed_vol_filter":
                vol_percentile = np.random.uniform(0.3, 0.7)

            try:
                equity = self.run_backtest_enhanced(
                    lookback_periods=base_lookback,
                    holding_period=holding_period,
                    n_stocks=n_stocks,
                    strategy=strategy,
                    vol_percentile=vol_percentile if vol_percentile else 0.5,
                    momentum_weight=momentum_weight if momentum_weight else 0.7,
                    use_momentum_weighting=np.random.choice([True, False]),
                )
                metrics = self.calculate_metrics(equity["portfolio_returns"])
                metrics["simulation"] = i
                metrics["n_stocks"] = n_stocks
                metrics["holding_period"] = holding_period
                if momentum_weight:
                    metrics["momentum_weight"] = momentum_weight
                results.append(metrics)
            except Exception as e:
                logging.warning(f"Simulation {i} failed: {e}")
                continue

            if (i + 1) % 20 == 0:
                logging.info(f"Completed {i + 1}/{n_simulations} simulations...")

        return pd.DataFrame(results)

    def bootstrap_simulation(
        self,
        lookback_periods: int,
        holding_period: int,
        n_stocks: int,
        strategy: str = "risk_adjusted_momentum",
        n_simulations: int = 100,
        block_size: int = 21,
        random_seed: int | None = None,
    ) -> pd.DataFrame:
        """Run bootstrap simulation by resampling return blocks.

        Tests strategy robustness under different return path scenarios
        by resampling blocks of returns with replacement.

        Args:
            lookback_periods: Lookback period for factor computation.
            holding_period: Holding period in days.
            n_stocks: Number of stocks to hold.
            strategy: Strategy to test.
            n_simulations: Number of bootstrap runs.
            block_size: Size of blocks to resample (in days).
            random_seed: Random seed for reproducibility.

        Returns:
            DataFrame with metrics for each bootstrap run.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        returns = self.factor_data.xs("returns", level=0, axis=1)
        n_days = len(returns)
        n_blocks = n_days // block_size

        results = []
        logging.info(f"Running {n_simulations} bootstrap simulations...")

        for i in range(n_simulations):
            try:
                # Create bootstrap sample by resampling blocks
                block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
                bootstrap_indices = []
                for block_idx in block_indices:
                    start = block_idx * block_size
                    end = min(start + block_size, n_days)
                    bootstrap_indices.extend(range(start, end))

                # Trim to original length
                bootstrap_indices = bootstrap_indices[:n_days]

                # Create bootstrap factor data
                bootstrap_factor_data = self.factor_data.iloc[bootstrap_indices].copy()
                bootstrap_factor_data.index = self.factor_data.index[: len(bootstrap_indices)]

                # Create temporary backtester with bootstrap data
                bootstrap_backtester = Backtester(
                    bootstrap_factor_data, initial_capital=self.initial_capital
                )

                equity = bootstrap_backtester.run_backtest_enhanced(
                    lookback_periods=lookback_periods,
                    holding_period=holding_period,
                    n_stocks=n_stocks,
                    strategy=strategy,
                )
                metrics = self.calculate_metrics(equity["portfolio_returns"])
                metrics["simulation"] = i
                results.append(metrics)
            except Exception as e:
                logging.warning(f"Bootstrap simulation {i} failed: {e}")
                continue

            if (i + 1) % 20 == 0:
                logging.info(f"Completed {i + 1}/{n_simulations} bootstrap runs...")

        return pd.DataFrame(results)

    def analyze_monte_carlo_results(self, mc_results: pd.DataFrame) -> pd.DataFrame:
        """Analyze Monte Carlo simulation results with statistics.

        Args:
            mc_results: DataFrame from monte_carlo_simulation() or bootstrap_simulation().

        Returns:
            DataFrame with summary statistics (mean, std, min, max, percentiles).
        """
        metric_cols = [
            "total_return",
            "annual_volatility",
            "max_drawdown",
            "annual_sharpe_ratio",
        ]
        available_metrics = [col for col in metric_cols if col in mc_results.columns]

        summary_stats = []
        for metric in available_metrics:
            values = mc_results[metric].dropna()
            if len(values) == 0:
                continue

            summary_stats.append(
                {
                    "metric": metric,
                    "mean": values.mean(),
                    "std": values.std(),
                    "min": values.min(),
                    "max": values.max(),
                    "median": values.median(),
                    "p5": values.quantile(0.05),  # 5th percentile
                    "p25": values.quantile(0.25),  # 25th percentile
                    "p75": values.quantile(0.75),  # 75th percentile
                    "p95": values.quantile(0.95),  # 95th percentile
                }
            )

        return pd.DataFrame(summary_stats)

    def plot_monte_carlo_distributions(
        self, mc_results: pd.DataFrame, title: str = "Monte Carlo Simulation Results"
    ) -> None:
        """Plot distributions of Monte Carlo simulation results.

        Args:
            mc_results: DataFrame from monte_carlo_simulation() or bootstrap_simulation().
            title: Plot title.
        """
        metric_cols = [
            "total_return",
            "annual_volatility",
            "max_drawdown",
            "annual_sharpe_ratio",
        ]
        available_metrics = [col for col in metric_cols if col in mc_results.columns]

        n_metrics = len(available_metrics)
        if n_metrics == 0:
            logging.warning("No metrics to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(available_metrics):
            if idx >= 4:
                break

            ax = axes[idx]
            values = mc_results[metric].dropna()

            # Histogram
            ax.hist(values, bins=30, alpha=0.7, edgecolor="black")
            ax.axvline(values.mean(), color="red", linestyle="--", linewidth=2, label="Mean")
            ax.axvline(values.median(), color="green", linestyle="--", linewidth=2, label="Median")
            ax.axvline(
                values.quantile(0.05),
                color="orange",
                linestyle=":",
                linewidth=1,
                label="5th/95th percentile",
            )
            ax.axvline(values.quantile(0.95), color="orange", linestyle=":", linewidth=1)

            ax.set_xlabel(metric.replace("_", " ").title())
            ax.set_ylabel("Frequency")
            ax.set_title(f"Distribution of {metric.replace('_', ' ').title()}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(len(available_metrics), 4):
            axes[idx].axis("off")

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def plot_monte_carlo_equity_curves(
        self,
        mc_results: pd.DataFrame,
        n_curves: int = 20,
        title: str = "Monte Carlo Equity Curves",
    ) -> None:
        """Plot sample equity curves from Monte Carlo simulations.

        Args:
            mc_results: DataFrame from monte_carlo_simulation() or bootstrap_simulation().
            n_curves: Number of sample curves to plot.
            title: Plot title.
        """
        # This requires storing equity curves during simulation
        # For now, we'll plot a simplified version showing return distributions
        if "total_return" not in mc_results.columns:
            logging.warning("Cannot plot equity curves without total_return metric")
            return

        plt.figure(figsize=(12, 6))
        returns = mc_results["total_return"].dropna()

        # Create a simple visualization of return distribution over time
        # (assuming simulations represent different time periods or scenarios)
        sorted_returns = np.sort(returns)
        cumulative_prob = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)

        plt.plot(sorted_returns, cumulative_prob * 100, linewidth=2)
        plt.axvline(returns.mean(), color="red", linestyle="--", linewidth=2, label="Mean")
        plt.axvline(returns.median(), color="green", linestyle="--", linewidth=2, label="Median")
        plt.fill_between(
            sorted_returns,
            0,
            cumulative_prob * 100,
            alpha=0.3,
            label="Confidence Band",
        )

        plt.xlabel("Total Return")
        plt.ylabel("Cumulative Probability (%)")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


class PredictiveMonteCarlo:
    """
    Predictive Monte Carlo simulation with explicit dependent and independent variables.
    
    This class implements a proper Monte Carlo model where:
    - Dependent variables: Portfolio performance metrics to predict
    - Independent variables: Market factors that drive performance
    - Probability distributions: Fitted from historical data
    - Simulations: Generate random values and predict outcomes
    """

    def __init__(self, factor_data: pd.DataFrame, backtester: Backtester):
        """Initialize predictive Monte Carlo model.

        Args:
            factor_data: Historical factor data (momentum, low_vol, returns).
            backtester: Backtester instance to use for predictions.
        """
        self.factor_data = factor_data
        self.backtester = backtester
        self.independent_vars: Dict[str, Dict] = {}
        self.dependent_vars: List[str] = [
            "total_return",
            "annual_sharpe_ratio",
            "max_drawdown",
            "annual_volatility",
        ]
        self.distributions: Dict[str, Dict] = {}

    def identify_variables(self) -> Dict[str, List[str]]:
        """Identify dependent and independent variables.

        Returns:
            Dictionary with 'dependent' and 'independent' variable lists.
        """
        # Independent variables (predictors/risk factors)
        independent = [
            "market_volatility",  # Overall market volatility regime
            "avg_momentum_strength",  # Average momentum in universe
            "momentum_spread",  # Difference between top and bottom momentum
            "avg_volatility",  # Average volatility in universe
            "market_correlation",  # Average correlation between stocks
            "market_trend",  # Bull/bear market indicator
        ]

        # Dependent variables (what we predict)
        dependent = self.dependent_vars

        return {"independent": independent, "dependent": dependent}

    def extract_market_factors(self) -> pd.DataFrame:
        """Extract independent variables from historical data.

        Returns:
            DataFrame with independent variables over time.
        """
        momentum = self.factor_data.xs("momentum", level=0, axis=1)
        low_vol = self.factor_data.xs("low_vol", level=0, axis=1)
        returns = self.factor_data.xs("returns", level=0, axis=1)

        # Market volatility: Rolling volatility of equal-weighted portfolio
        market_returns = returns.mean(axis=1)
        market_volatility = (
            market_returns.rolling(window=TRADING_DAYS_PER_YEAR, min_periods=60)
            .std()
            .multiply(np.sqrt(TRADING_DAYS_PER_YEAR))
        )

        # Average momentum strength: Mean momentum across all stocks
        avg_momentum_strength = momentum.mean(axis=1)

        # Momentum spread: Difference between top and bottom quartile
        momentum_75 = momentum.quantile(0.75, axis=1)
        momentum_25 = momentum.quantile(0.25, axis=1)
        momentum_spread = momentum_75 - momentum_25

        # Average volatility: Mean volatility across all stocks
        avg_volatility = low_vol.mean(axis=1)

        # Market correlation: Average pairwise correlation of returns
        # Use rolling window to compute correlation
        rolling_corr = []
        window = 60  # ~3 months
        for i in range(window, len(returns)):
            window_returns = returns.iloc[i - window : i]
            corr_matrix = window_returns.corr()
            # Average correlation (excluding diagonal)
            avg_corr = (
                corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            )
            rolling_corr.append(avg_corr)

        # Pad with NaN for initial period
        market_correlation = pd.Series(
            [np.nan] * window + rolling_corr, index=returns.index
        )

        # Market trend: Bull (1) or Bear (-1) based on recent returns
        market_trend = pd.Series(
            np.where(market_returns.rolling(60).mean() > 0, 1, -1), index=returns.index
        )

        factors = pd.DataFrame(
            {
                "market_volatility": market_volatility,
                "avg_momentum_strength": avg_momentum_strength,
                "momentum_spread": momentum_spread,
                "avg_volatility": avg_volatility,
                "market_correlation": market_correlation,
                "market_trend": market_trend,
            }
        )

        return factors.dropna()

    def fit_probability_distributions(
        self, factors: pd.DataFrame, distribution_type: str = "normal"
    ) -> Dict[str, Dict]:
        """Fit probability distributions to independent variables.

        Args:
            factors: DataFrame with independent variables.
            distribution_type: Type of distribution ('normal', 'lognormal', 'empirical').

        Returns:
            Dictionary with distribution parameters for each variable.
        """
        distributions = {}

        for var in factors.columns:
            values = factors[var].dropna()

            if distribution_type == "normal":
                # Fit normal distribution: N(mean, std)
                mean = values.mean()
                std = values.std()
                distributions[var] = {
                    "type": "normal",
                    "mean": mean,
                    "std": std,
                    "min": values.min(),
                    "max": values.max(),
                }

            elif distribution_type == "lognormal":
                # Fit log-normal distribution (for positive variables)
                if (values > 0).all():
                    log_values = np.log(values)
                    mean_log = log_values.mean()
                    std_log = log_values.std()
                    distributions[var] = {
                        "type": "lognormal",
                        "mean_log": mean_log,
                        "std_log": std_log,
                        "min": values.min(),
                        "max": values.max(),
                    }
                else:
                    # Fall back to normal if not all positive
                    mean = values.mean()
                    std = values.std()
                    distributions[var] = {
                        "type": "normal",
                        "mean": mean,
                        "std": std,
                        "min": values.min(),
                        "max": values.max(),
                    }

            elif distribution_type == "empirical":
                # Use empirical distribution (histogram-based)
                distributions[var] = {
                    "type": "empirical",
                    "values": values.values,
                    "min": values.min(),
                    "max": values.max(),
                }

        self.distributions = distributions
        return distributions

    def sample_independent_variables(self, n_samples: int = 1) -> pd.DataFrame:
        """Generate random samples from probability distributions.

        Args:
            n_samples: Number of samples to generate.

        Returns:
            DataFrame with sampled independent variables.
        """
        samples = {}

        for var, dist_params in self.distributions.items():
            if dist_params["type"] == "normal":
                samples[var] = np.random.normal(
                    dist_params["mean"], dist_params["std"], n_samples
                )
                # Clip to historical range
                samples[var] = np.clip(
                    samples[var], dist_params["min"], dist_params["max"]
                )

            elif dist_params["type"] == "lognormal":
                log_samples = np.random.normal(
                    dist_params["mean_log"], dist_params["std_log"], n_samples
                )
                samples[var] = np.exp(log_samples)
                samples[var] = np.clip(
                    samples[var], dist_params["min"], dist_params["max"]
                )

            elif dist_params["type"] == "empirical":
                # Sample with replacement from historical values
                samples[var] = np.random.choice(
                    dist_params["values"], size=n_samples, replace=True
                )

        return pd.DataFrame(samples)

    def build_predictive_model(self, factors: pd.DataFrame) -> Dict:
        """Build predictive model linking independent to dependent variables.

        Uses historical backtest results to establish relationships.

        Args:
            factors: DataFrame with independent variables.

        Returns:
            Dictionary with model parameters/coefficients.
        """
        # Run backtest on historical data to get dependent variables
        # We'll use rolling windows to build the model
        logging.info("Building predictive model from historical data...")

        # Sample periods from history and run mini-backtests
        window_size = TRADING_DAYS_PER_YEAR * 2  # 2 years
        step_size = TRADING_DAYS_PER_YEAR // 4  # ~3 months

        X_samples = []  # Independent variables
        y_samples = []  # Dependent variables

        returns = self.factor_data.xs("returns", level=0, axis=1)
        n_periods = len(returns) - window_size

        for i in range(0, n_periods, step_size):
            try:
                # Get factor snapshot at start of period
                period_start = returns.index[i]
                period_end = returns.index[min(i + window_size, len(returns) - 1)]

                # Extract independent variables at period start
                if period_start in factors.index:
                    X_sample = factors.loc[period_start].values
                    X_samples.append(X_sample)

                    # Run mini-backtest for this period
                    period_factor_data = self.factor_data.loc[period_start:period_end]
                    period_backtester = Backtester(
                        period_factor_data, initial_capital=self.backtester.initial_capital
                    )

                    equity = period_backtester.run_backtest_enhanced(
                        lookback_periods=TRADING_DAYS_PER_YEAR,
                        holding_period=21,
                        n_stocks=5,
                        strategy="risk_adjusted_momentum",
                    )

                    metrics = period_backtester.calculate_metrics(
                        equity["portfolio_returns"]
                    )
                    y_sample = [metrics[var] for var in self.dependent_vars]
                    y_samples.append(y_sample)

            except Exception as e:
                logging.debug(f"Skipping period {i}: {e}")
                continue

        if len(X_samples) == 0:
            raise ValueError("Could not build predictive model - no valid samples")

        # Store model data for prediction
        self.model_data = {
            "X_samples": np.array(X_samples),
            "y_samples": np.array(y_samples),
            "feature_names": factors.columns.tolist(),
            "target_names": self.dependent_vars,
        }

        logging.info(f"Built model with {len(X_samples)} historical samples")
        return self.model_data

    def predict_from_factors(
        self, independent_vars: pd.DataFrame, method: str = "nearest_neighbor"
    ) -> pd.DataFrame:
        """Predict dependent variables from independent variables.

        Args:
            independent_vars: DataFrame with independent variable values.
            method: Prediction method ('nearest_neighbor', 'linear', 'average').

        Returns:
            DataFrame with predicted dependent variables.
        """
        if not hasattr(self, "model_data"):
            raise ValueError("Must call build_predictive_model() first")

        X_samples = self.model_data["X_samples"]
        y_samples = self.model_data["y_samples"]
        predictions = []

        for _, row in independent_vars.iterrows():
            x = row.values

            if method == "nearest_neighbor":
                # Find k nearest neighbors in historical data
                k = min(5, len(X_samples))
                distances = np.sqrt(((X_samples - x) ** 2).sum(axis=1))
                nearest_indices = np.argsort(distances)[:k]

                # Weighted average of nearest neighbors (inverse distance weighting)
                weights = 1 / (distances[nearest_indices] + 1e-8)
                weights = weights / weights.sum()

                prediction = (y_samples[nearest_indices] * weights[:, np.newaxis]).sum(
                    axis=0
                )

            elif method == "linear":
                # Simple linear regression (OLS)
                try:
                    from sklearn.linear_model import LinearRegression

                    model = LinearRegression()
                    model.fit(X_samples, y_samples)
                    prediction = model.predict(x.reshape(1, -1))[0]
                except ImportError:
                    logging.warning(
                        "scikit-learn not available, falling back to nearest_neighbor"
                    )
                    # Fall back to nearest neighbor
                    k = min(5, len(X_samples))
                    distances = np.sqrt(((X_samples - x) ** 2).sum(axis=1))
                    nearest_indices = np.argsort(distances)[:k]
                    weights = 1 / (distances[nearest_indices] + 1e-8)
                    weights = weights / weights.sum()
                    prediction = (y_samples[nearest_indices] * weights[:, np.newaxis]).sum(
                        axis=0
                    )

            elif method == "average":
                # Simple average of all historical outcomes
                prediction = y_samples.mean(axis=0)

            predictions.append(prediction)

        predictions_df = pd.DataFrame(
            predictions, columns=self.dependent_vars, index=independent_vars.index
        )
        return predictions_df

    def run_predictive_monte_carlo(
        self,
        n_simulations: int = 1000,
        distribution_type: str = "normal",
        prediction_method: str = "nearest_neighbor",
        random_seed: int | None = None,
    ) -> pd.DataFrame:
        """Run full predictive Monte Carlo simulation.

        Steps:
        1. Extract independent variables from historical data
        2. Fit probability distributions
        3. Build predictive model
        4. Generate random samples from distributions
        5. Predict dependent variables
        6. Collect results

        Args:
            n_simulations: Number of Monte Carlo runs.
            distribution_type: Type of distribution to fit.
            prediction_method: Method for predicting dependent variables.
            random_seed: Random seed for reproducibility.

        Returns:
            DataFrame with predicted dependent variables for each simulation.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        logging.info("=" * 80)
        logging.info("PREDICTIVE MONTE CARLO SIMULATION")
        logging.info("=" * 80)

        # Step 1: Extract independent variables
        logging.info("Step 1: Extracting independent variables from historical data...")
        factors = self.extract_market_factors()
        logging.info(f"Extracted {len(factors)} observations of {len(factors.columns)} factors")

        # Step 2: Fit probability distributions
        logging.info(f"Step 2: Fitting {distribution_type} distributions...")
        distributions = self.fit_probability_distributions(factors, distribution_type)
        logging.info(f"Fitted distributions for {len(distributions)} variables")

        # Step 3: Build predictive model
        logging.info("Step 3: Building predictive model from historical backtests...")
        model_data = self.build_predictive_model(factors)
        logging.info("Model built successfully")

        # Step 4 & 5: Generate samples and predict
        logging.info(f"Step 4-5: Running {n_simulations} simulations...")
        results = []

        for i in range(n_simulations):
            # Generate random independent variables
            independent_sample = self.sample_independent_variables(n_samples=1)

            # Predict dependent variables
            predictions = self.predict_from_factors(
                independent_sample, method=prediction_method
            )

            # Store results
            result = predictions.iloc[0].to_dict()
            result["simulation"] = i
            # Also store the independent variables used
            for var in independent_sample.columns:
                result[f"input_{var}"] = independent_sample[var].iloc[0]
            results.append(result)

            if (i + 1) % 100 == 0:
                logging.info(f"Completed {i + 1}/{n_simulations} simulations...")

        results_df = pd.DataFrame(results)
        logging.info("Monte Carlo simulation completed")
        return results_df


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

    # Compare all enhanced strategies
    print("=" * 80)
    print("STRATEGY COMPARISON")
    print("=" * 80)
    comparison = backtester.compare_strategies(
        lookback_periods=TRADING_DAYS_PER_YEAR, holding_period=21, n_stocks=5
    )
    print("\nStrategy Performance Metrics:")
    print(comparison.to_string(index=False))
    print("\n" + "=" * 80)

    # Test the best-performing strategy (typically risk_adjusted_momentum or composite_score)
    print("\nRunning Enhanced Strategy: risk_adjusted_momentum")
    print("=" * 80)
    equity_enhanced = backtester.run_backtest_enhanced(
        lookback_periods=TRADING_DAYS_PER_YEAR,
        holding_period=21,
        n_stocks=5,
        strategy="risk_adjusted_momentum",
        use_momentum_weighting=False,
    )
    metrics_enhanced = backtester.calculate_metrics(equity_enhanced["portfolio_returns"])

    print("\nEnhanced Strategy Performance:")
    for key, value in metrics_enhanced.items():
        print(f"{key}: {value:.4f}")

    # Compare with original strategy
    print("\n" + "=" * 80)
    print("Original Strategy (Low Vol Filter + Momentum):")
    equity_original = backtester.run_backtest(
        lookback_periods=TRADING_DAYS_PER_YEAR, holding_period=21, n_stocks=5
    )
    metrics_original = backtester.calculate_metrics(equity_original["portfolio_returns"])
    for key, value in metrics_original.items():
        print(f"{key}: {value:.4f}")

    print("\n" + "=" * 80)
    print("Improvement:")
    improvement = {
        k: metrics_enhanced[k] - metrics_original[k]
        for k in metrics_enhanced.keys()
    }
    for key, value in improvement.items():
        pct = (value / abs(metrics_original[key]) * 100) if metrics_original[key] != 0 else 0
        print(f"{key}: {value:+.4f} ({pct:+.2f}%)")

    # Plot enhanced strategy
    backtester.plot_results(
        equity_enhanced[["portfolio", "benchmark"]],
        title="Enhanced Strategy: Risk-Adjusted Momentum vs Benchmark"
    )

    # Monte Carlo Simulation
    print("\n" + "=" * 80)
    print("MONTE CARLO SIMULATION")
    print("=" * 80)
    print("Testing strategy robustness with parameter randomization...")

    mc_results = backtester.monte_carlo_simulation(
        base_lookback=TRADING_DAYS_PER_YEAR,
        base_holding_period=21,
        base_n_stocks=5,
        strategy="risk_adjusted_momentum",
        n_simulations=100,
        param_ranges={
            "n_stocks": (3, 8),
            "holding_period": (15, 30),
            "momentum_weight": (0.5, 0.9),
        },
        random_seed=42,
    )

    print(f"\nCompleted {len(mc_results)} successful simulations")
    print("\nMonte Carlo Summary Statistics:")
    print("=" * 80)
    mc_summary = backtester.analyze_monte_carlo_results(mc_results)
    print(mc_summary.to_string(index=False))

    print("\n" + "=" * 80)
    print("Key Insights:")
    print("=" * 80)
    if "total_return" in mc_summary["metric"].values:
        tr_row = mc_summary[mc_summary["metric"] == "total_return"].iloc[0]
        print(f"Total Return:")
        print(f"  Mean: {tr_row['mean']:.4f}")
        print(f"  95% Confidence Interval: [{tr_row['p5']:.4f}, {tr_row['p95']:.4f}]")
        print(f"  Range: [{tr_row['min']:.4f}, {tr_row['max']:.4f}]")

    if "annual_sharpe_ratio" in mc_summary["metric"].values:
        sharpe_row = mc_summary[mc_summary["metric"] == "annual_sharpe_ratio"].iloc[0]
        print(f"\nSharpe Ratio:")
        print(f"  Mean: {sharpe_row['mean']:.4f}")
        print(f"  95% Confidence Interval: [{sharpe_row['p5']:.4f}, {sharpe_row['p95']:.4f}]")

    if "max_drawdown" in mc_summary["metric"].values:
        mdd_row = mc_summary[mc_summary["metric"] == "max_drawdown"].iloc[0]
        print(f"\nMax Drawdown:")
        print(f"  Mean: {mdd_row['mean']:.4f}")
        print(f"  95% Confidence Interval: [{mdd_row['p5']:.4f}, {mdd_row['p95']:.4f}]")

    # Plot Monte Carlo distributions
    backtester.plot_monte_carlo_distributions(
        mc_results, title="Monte Carlo Simulation: Performance Distributions"
    )

    # Plot return distribution
    backtester.plot_monte_carlo_equity_curves(
        mc_results, title="Monte Carlo: Total Return Distribution"
    )

    # Predictive Monte Carlo Simulation
    print("\n" + "=" * 80)
    print("PREDICTIVE MONTE CARLO SIMULATION")
    print("=" * 80)
    print("Using predictive model with dependent and independent variables...")

    predictive_mc = PredictiveMonteCarlo(factors, backtester)

    # Identify variables
    variables = predictive_mc.identify_variables()
    print("\nModel Structure:")
    print(f"  Independent Variables (Predictors): {', '.join(variables['independent'])}")
    print(f"  Dependent Variables (Predictions): {', '.join(variables['dependent'])}")

    # Run predictive Monte Carlo
    try:
        predictive_results = predictive_mc.run_predictive_monte_carlo(
            n_simulations=500,
            distribution_type="normal",
            prediction_method="nearest_neighbor",
            random_seed=42,
        )

        print(f"\nCompleted {len(predictive_results)} predictive simulations")
        print("\nPredictive Monte Carlo Summary:")
        print("=" * 80)

        # Analyze predicted dependent variables
        pred_summary = backtester.analyze_monte_carlo_results(
            predictive_results[variables["dependent"]]
        )
        print(pred_summary.to_string(index=False))

        print("\n" + "=" * 80)
        print("Predictive Model Insights:")
        print("=" * 80)
        if "total_return" in pred_summary["metric"].values:
            tr_row = pred_summary[pred_summary["metric"] == "total_return"].iloc[0]
            print(f"Predicted Total Return:")
            print(f"  Mean: {tr_row['mean']:.4f}")
            print(f"  95% Confidence Interval: [{tr_row['p5']:.4f}, {tr_row['p95']:.4f}]")
            print(f"  Range: [{tr_row['min']:.4f}, {tr_row['max']:.4f}]")

        # Show distribution of independent variables used
        print("\nIndependent Variable Ranges (from simulations):")
        for var in variables["independent"]:
            input_col = f"input_{var}"
            if input_col in predictive_results.columns:
                values = predictive_results[input_col]
                print(
                    f"  {var}: [{values.min():.4f}, {values.max():.4f}], "
                    f"mean={values.mean():.4f}"
                )

        # Plot predictive results
        backtester.plot_monte_carlo_distributions(
            predictive_results[variables["dependent"]],
            title="Predictive Monte Carlo: Predicted Performance Distributions",
        )

    except Exception as e:
        logging.error(f"Predictive Monte Carlo failed: {e}")
        print(f"\nPredictive Monte Carlo encountered an error: {e}")
        print("This may require more historical data or different parameters.")

