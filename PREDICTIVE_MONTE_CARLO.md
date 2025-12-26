# Predictive Monte Carlo Simulation

## Overview

This implementation follows the traditional Monte Carlo methodology with explicit **dependent** and **independent** variables, probability distributions, and predictive modeling.

## Model Structure

### Dependent Variables (What We Predict)

The model predicts portfolio performance metrics:

1. **`total_return`**: Cumulative portfolio return
2. **`annual_sharpe_ratio`**: Risk-adjusted return measure
3. **`max_drawdown`**: Worst peak-to-trough decline
4. **`annual_volatility`**: Portfolio risk measure

### Independent Variables (Predictors/Risk Factors)

Market factors that drive portfolio performance:

1. **`market_volatility`**: Overall market volatility regime
   - Rolling 1-year volatility of equal-weighted portfolio
   - Measures market stress/uncertainty

2. **`avg_momentum_strength`**: Average momentum across all stocks
   - Mean momentum factor value in universe
   - Indicates overall market trend strength

3. **`momentum_spread`**: Difference between top and bottom momentum quartiles
   - Measures dispersion in momentum signals
   - Higher spread = more selection opportunity

4. **`avg_volatility`**: Average stock volatility in universe
   - Mean low_vol factor across all stocks
   - Measures overall market risk level

5. **`market_correlation`**: Average pairwise correlation between stocks
   - Rolling correlation of stock returns
   - Higher correlation = less diversification benefit

6. **`market_trend`**: Bull/bear market indicator
   - Binary: +1 (bull) or -1 (bear)
   - Based on recent market returns

## Methodology

### Step 1: Extract Independent Variables

From historical data, we compute each independent variable over time:

```python
factors = predictive_mc.extract_market_factors()
# Returns DataFrame with 6 columns × N time periods
```

**Example**:
- `market_volatility`: 0.15 (15% annualized)
- `avg_momentum_strength`: 0.12 (12% average momentum)
- `momentum_spread`: 0.35 (35% spread)
- `avg_volatility`: 0.22 (22% average vol)
- `market_correlation`: 0.45 (45% average correlation)
- `market_trend`: 1 (bull market)

### Step 2: Fit Probability Distributions

For each independent variable, we fit a probability distribution from historical data:

**Distribution Types**:
- **Normal**: `N(μ, σ)` - Fitted to mean and std
- **Log-normal**: For positive-only variables
- **Empirical**: Histogram-based (sample from historical values)

**Example**:
```python
distributions = {
    'market_volatility': {
        'type': 'normal',
        'mean': 0.18,
        'std': 0.06,
        'min': 0.08,
        'max': 0.35
    },
    'avg_momentum_strength': {
        'type': 'normal',
        'mean': 0.10,
        'std': 0.15,
        'min': -0.30,
        'max': 0.50
    },
    ...
}
```

### Step 3: Build Predictive Model

The model learns the relationship between independent and dependent variables:

**Process**:
1. Sample rolling windows from historical data (e.g., 2-year periods)
2. Extract independent variables at start of each period
3. Run mini-backtest for that period
4. Collect dependent variables (performance metrics)
5. Build model linking X (independent) → Y (dependent)

**Model Types**:
- **Nearest Neighbor**: Find similar historical periods, weighted average
- **Linear Regression**: OLS model (requires scikit-learn)
- **Average**: Simple average of all historical outcomes

### Step 4: Generate Random Samples

For each simulation:
1. Sample random values from probability distributions
2. Each independent variable gets a random value
3. Creates a "scenario" of market conditions

**Example Simulation**:
```python
Simulation 1:
  market_volatility: 0.20 (sampled from N(0.18, 0.06))
  avg_momentum_strength: 0.08 (sampled from N(0.10, 0.15))
  momentum_spread: 0.42 (sampled from distribution)
  ...
```

### Step 5: Predict Dependent Variables

Using the predictive model:
1. Input: Random independent variable values
2. Model finds similar historical periods
3. Predicts dependent variables based on historical relationships

**Example Prediction**:
```python
Input (Independent):
  market_volatility: 0.20
  avg_momentum_strength: 0.08
  ...

Output (Dependent - Predicted):
  total_return: 0.45 (45%)
  annual_sharpe_ratio: 1.2
  max_drawdown: -0.15 (-15%)
  annual_volatility: 0.18 (18%)
```

### Step 6: Collect Results

After N simulations (e.g., 1000), you have:
- N predictions for each dependent variable
- Distribution of outcomes
- Statistical analysis (mean, std, percentiles)

## Usage

### Basic Example

```python
from MultiFactorBacktester import PredictiveMonteCarlo

# Initialize
predictive_mc = PredictiveMonteCarlo(factor_data, backtester)

# Identify variables
variables = predictive_mc.identify_variables()
print(f"Independent: {variables['independent']}")
print(f"Dependent: {variables['dependent']}")

# Run simulation
results = predictive_mc.run_predictive_monte_carlo(
    n_simulations=1000,
    distribution_type="normal",
    prediction_method="nearest_neighbor",
    random_seed=42
)

# Analyze results
summary = backtester.analyze_monte_carlo_results(
    results[variables['dependent']])
print(summary)
```

### Customization

#### Distribution Types

```python
# Normal distribution (default)
results = predictive_mc.run_predictive_monte_carlo(
    distribution_type="normal"
)

# Log-normal (for positive variables)
results = predictive_mc.run_predictive_monte_carlo(
    distribution_type="lognormal"
)

# Empirical (sample from historical values)
results = predictive_mc.run_predictive_monte_carlo(
    distribution_type="empirical"
)
```

#### Prediction Methods

```python
# Nearest neighbor (default, no dependencies)
results = predictive_mc.run_predictive_monte_carlo(
    prediction_method="nearest_neighbor"
)

# Linear regression (requires scikit-learn)
results = predictive_mc.run_predictive_monte_carlo(
    prediction_method="linear"
)

# Simple average
results = predictive_mc.run_predictive_monte_carlo(
    prediction_method="average"
)
```

## Interpretation

### What the Model Tells You

**Example Output**:
```
Predicted Total Return:
  Mean: 0.4523 (45.23%)
  95% CI: [0.1234, 0.7891]  (12.34% to 78.91%)
  Range: [0.0123, 0.9123]  (1.23% to 91.23%)
```

**Interpretation**:
- **Expected return**: 45.23% (mean prediction)
- **Confidence**: 95% of scenarios predict returns between 12% and 79%
- **Worst case**: 1.23% (still positive!)
- **Best case**: 91.23%

### Understanding Independent Variables

The model shows which market conditions drive performance:

```python
# High market volatility → Lower returns?
# High momentum spread → Better selection → Higher returns?
# Bull market → Better performance?
```

You can analyze correlations:
```python
correlations = results[['input_market_volatility', 'total_return']].corr()
print(correlations)
```

## Advantages Over Parameter-Based Monte Carlo

### Parameter-Based (Previous Method)
- Tests: "What if I used different parameters?"
- Varies: n_stocks, holding_period, etc.
- **Limitation**: Assumes same market conditions

### Predictive Monte Carlo (New Method)
- Tests: "What if market conditions were different?"
- Varies: Market volatility, momentum, correlations, etc.
- **Advantage**: Tests strategy under different market regimes

## Model Validation

### Check Model Quality

1. **Historical Fit**: Compare predicted vs actual on historical data
2. **Sensitivity Analysis**: Vary one independent variable, see impact
3. **Out-of-Sample**: Test on periods not used in training

### Example Validation

```python
# Extract factors for a specific period
test_factors = predictive_mc.extract_market_factors()

# Predict for that period
predictions = predictive_mc.predict_from_factors(
    test_factors.iloc[[0]],  # One period
    method="nearest_neighbor"
)

# Compare with actual backtest result
actual = backtester.calculate_metrics(...)
print(f"Predicted: {predictions.iloc[0]['total_return']:.4f}")
print(f"Actual: {actual['total_return']:.4f}")
```

## Best Practices

### 1. Number of Simulations
- **500-1000**: Good balance for most cases
- **1000+**: For high precision
- More simulations = smoother distributions

### 2. Distribution Type
- **Normal**: Good default, works for most variables
- **Log-normal**: Use for strictly positive variables (volatility)
- **Empirical**: Use when distribution is non-normal

### 3. Prediction Method
- **Nearest Neighbor**: Robust, no dependencies, good default
- **Linear**: Fast, but assumes linear relationships
- **Average**: Simple baseline, less accurate

### 4. Model Building
- Ensure sufficient historical data (at least 2-3 years)
- Use rolling windows to capture different market regimes
- Validate on out-of-sample periods

## Limitations

1. **Historical Relationships**: Assumes past relationships hold in future
2. **Stationarity**: Market dynamics may change over time
3. **Model Complexity**: Simple models may miss non-linear relationships
4. **Data Requirements**: Needs sufficient history to build reliable model

## Next Steps

1. **Add More Factors**: Include macroeconomic variables, sentiment, etc.
2. **Non-linear Models**: Use machine learning (random forest, neural networks)
3. **Regime Detection**: Model different market regimes separately
4. **Dynamic Distributions**: Update distributions over time
5. **Stress Testing**: Test extreme scenarios (tail events)

