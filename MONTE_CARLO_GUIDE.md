# Monte Carlo Simulation Guide

## Overview

The Monte Carlo simulation framework tests strategy robustness by running multiple backtests with randomized parameters or resampled data. This helps assess:

- **Parameter Sensitivity**: How performance varies with different parameter settings
- **Strategy Robustness**: Whether results are consistent across different scenarios
- **Confidence Intervals**: Expected range of performance outcomes
- **Risk Assessment**: Worst-case and best-case scenarios

## Methods

### 1. Parameter-Based Monte Carlo (`monte_carlo_simulation`)

Tests strategy robustness by varying parameters within specified ranges.

**How it works**:
- Randomly samples parameters (n_stocks, holding_period, momentum_weight, etc.)
- Runs backtest for each parameter combination
- Collects performance metrics across all runs
- Provides statistical analysis

**Usage**:
```python
mc_results = backtester.monte_carlo_simulation(
    base_lookback=252,
    base_holding_period=21,
    base_n_stocks=5,
    strategy="risk_adjusted_momentum",
    n_simulations=100,
    param_ranges={
        "n_stocks": (3, 8),           # Range for number of stocks
        "holding_period": (15, 30),    # Range for holding period
        "momentum_weight": (0.5, 0.9), # For composite_score strategy
    },
    random_seed=42,  # For reproducibility
)
```

**Parameters**:
- `n_simulations`: Number of Monte Carlo runs (100-1000 recommended)
- `param_ranges`: Dictionary specifying parameter ranges
- `random_seed`: Set for reproducible results

### 2. Bootstrap Simulation (`bootstrap_simulation`)

Tests strategy robustness under different return path scenarios by resampling blocks of historical returns.

**How it works**:
- Resamples blocks of returns with replacement
- Creates alternative return paths
- Tests strategy on these alternative scenarios
- Assesses performance under different market conditions

**Usage**:
```python
bootstrap_results = backtester.bootstrap_simulation(
    lookback_periods=252,
    holding_period=21,
    n_stocks=5,
    strategy="risk_adjusted_momentum",
    n_simulations=100,
    block_size=21,  # Size of blocks to resample (in days)
    random_seed=42,
)
```

**When to use**:
- Test strategy under different market regimes
- Assess performance if historical patterns were different
- Understand path dependency

## Analysis Methods

### 1. Statistical Summary (`analyze_monte_carlo_results`)

Provides comprehensive statistics for all metrics:

```python
summary = backtester.analyze_monte_carlo_results(mc_results)
print(summary)
```

**Output includes**:
- Mean, median, standard deviation
- Min, max values
- Percentiles (5th, 25th, 75th, 95th)
- Confidence intervals

**Metrics analyzed**:
- `total_return`: Cumulative return
- `annual_volatility`: Risk measure
- `max_drawdown`: Worst peak-to-trough decline
- `annual_sharpe_ratio`: Risk-adjusted return

### 2. Distribution Visualization (`plot_monte_carlo_distributions`)

Plots histograms showing the distribution of each performance metric:

```python
backtester.plot_monte_carlo_distributions(
    mc_results,
    title="Monte Carlo Simulation Results"
)
```

**Features**:
- Histograms for each metric
- Mean and median lines
- 5th and 95th percentile markers
- Visual assessment of distribution shape

### 3. Return Distribution (`plot_monte_carlo_equity_curves`)

Shows cumulative probability distribution of returns:

```python
backtester.plot_monte_carlo_equity_curves(
    mc_results,
    title="Return Distribution"
)
```

**Use for**:
- Understanding probability of different return outcomes
- Assessing downside risk
- Comparing strategies

## Interpreting Results

### Key Statistics to Focus On

1. **Mean vs Median**:
   - If mean > median: Positive skew (some very good outcomes)
   - If mean < median: Negative skew (some very bad outcomes)
   - Close values: Symmetric distribution

2. **Confidence Intervals (5th-95th percentile)**:
   - Shows expected range of outcomes
   - Narrow interval = more consistent results
   - Wide interval = high uncertainty

3. **Standard Deviation**:
   - Low std = robust strategy (consistent results)
   - High std = sensitive to parameters (unstable)

4. **Min/Max**:
   - Worst-case scenario (min)
   - Best-case scenario (max)
   - Assess tail risks

### Example Interpretation

```
Total Return:
  Mean: 0.4523
  95% Confidence Interval: [0.1234, 0.7891]
  Range: [0.0123, 0.9123]
```

**Interpretation**:
- Average return is 45.23%
- 95% of runs fall between 12.34% and 78.91%
- Worst case: 1.23% (still positive!)
- Best case: 91.23%

**Assessment**: Strategy is robust with positive returns in most scenarios.

## Best Practices

### 1. Number of Simulations
- **100 simulations**: Quick test, ~1-2 minutes
- **500 simulations**: Good balance, ~5-10 minutes
- **1000+ simulations**: Thorough analysis, ~15-30 minutes

### 2. Parameter Ranges
- Start with wide ranges to understand sensitivity
- Narrow ranges around optimal values for fine-tuning
- Consider realistic bounds (e.g., n_stocks: 3-10, not 1-30)

### 3. Multiple Strategies
Run Monte Carlo for each strategy and compare:
```python
strategies = ["risk_adjusted_momentum", "composite_score", "momentum_first"]
for strategy in strategies:
    mc_results = backtester.monte_carlo_simulation(
        strategy=strategy,
        n_simulations=100,
        ...
    )
    # Compare results
```

### 4. Reproducibility
Always set `random_seed` for reproducible results:
```python
random_seed=42  # Or any integer
```

## Example Workflow

```python
# 1. Run Monte Carlo simulation
mc_results = backtester.monte_carlo_simulation(
    base_lookback=252,
    base_holding_period=21,
    base_n_stocks=5,
    strategy="risk_adjusted_momentum",
    n_simulations=200,
    param_ranges={
        "n_stocks": (3, 8),
        "holding_period": (15, 30),
    },
    random_seed=42,
)

# 2. Analyze results
summary = backtester.analyze_monte_carlo_results(mc_results)
print(summary)

# 3. Visualize distributions
backtester.plot_monte_carlo_distributions(mc_results)

# 4. Check key metrics
if "total_return" in summary["metric"].values:
    tr_row = summary[summary["metric"] == "total_return"].iloc[0]
    print(f"Expected return: {tr_row['mean']:.2%}")
    print(f"95% CI: [{tr_row['p5']:.2%}, {tr_row['p95']:.2%}]")
```

## Common Use Cases

### 1. Parameter Optimization
Find optimal parameter ranges:
```python
# Test different n_stocks ranges
for n_range in [(3, 5), (5, 7), (7, 10)]:
    mc_results = backtester.monte_carlo_simulation(
        param_ranges={"n_stocks": n_range},
        ...
    )
    # Compare mean Sharpe ratios
```

### 2. Strategy Comparison
Compare robustness across strategies:
```python
strategies = ["risk_adjusted_momentum", "composite_score"]
for strategy in strategies:
    mc_results = backtester.monte_carlo_simulation(strategy=strategy, ...)
    # Compare consistency (std) and mean performance
```

### 3. Risk Assessment
Assess downside risk:
```python
mc_results = backtester.monte_carlo_simulation(...)
summary = backtester.analyze_monte_carlo_results(mc_results)

# Check worst-case scenarios
mdd_row = summary[summary["metric"] == "max_drawdown"].iloc[0]
print(f"Worst drawdown: {mdd_row['min']:.2%}")
print(f"95% of runs have drawdown < {mdd_row['p95']:.2%}")
```

## Troubleshooting

### Issue: Simulations failing
- **Cause**: Invalid parameter combinations
- **Solution**: Adjust parameter ranges, check for edge cases

### Issue: All results similar
- **Cause**: Parameter ranges too narrow
- **Solution**: Widen ranges to see more variation

### Issue: Very wide confidence intervals
- **Cause**: Strategy highly sensitive to parameters
- **Solution**: Consider more robust strategy or narrower parameter ranges

### Issue: Slow execution
- **Cause**: Too many simulations or complex strategy
- **Solution**: Reduce n_simulations or optimize strategy code

