# What Monte Carlo Simulation Does in This Backtester

## The Core Question

**"What if I had chosen different parameters? Would my strategy still work?"**

Monte Carlo simulation answers this by testing your strategy **hundreds of times** with **randomly varied parameters** to see how robust it is.

---

## Method 1: Parameter-Based Monte Carlo (`monte_carlo_simulation`)

### What It Actually Does (Step-by-Step)

#### Step 1: Randomize Parameters
For each of 100 simulations, it randomly picks:
- **n_stocks**: Randomly between 3-8 (instead of fixed 5)
- **holding_period**: Randomly between 15-30 days (instead of fixed 21)
- **momentum_weight**: Randomly between 0.5-0.9 (if using composite_score)
- **use_momentum_weighting**: Randomly True or False

**Example of 3 simulations:**
```
Simulation 1: n_stocks=4, holding_period=18, momentum_weight=0.65, weighting=False
Simulation 2: n_stocks=7, holding_period=28, momentum_weight=0.82, weighting=True
Simulation 3: n_stocks=5, holding_period=22, momentum_weight=0.71, weighting=False
```

#### Step 2: Run Full Backtest
For each random parameter set, it runs the **complete backtest**:
- Uses the **same historical data** (2010-2025)
- Applies the strategy with those random parameters
- Calculates all performance metrics (return, Sharpe, drawdown, etc.)

#### Step 3: Collect Results
After 100 runs, you have 100 different performance outcomes:
```
Simulation 1: total_return=0.45, sharpe=1.2, max_drawdown=-0.15
Simulation 2: total_return=0.38, sharpe=1.1, max_drawdown=-0.18
Simulation 3: total_return=0.52, sharpe=1.3, max_drawdown=-0.12
...
Simulation 100: total_return=0.41, sharpe=1.15, max_drawdown=-0.16
```

#### Step 4: Statistical Analysis
The code calculates statistics across all 100 runs:
- **Mean**: Average performance (e.g., average return = 0.43)
- **Std**: How much results vary (low std = consistent, high std = unstable)
- **Percentiles**: 
  - 5th percentile: Worst 5% of outcomes
  - 95th percentile: Best 5% of outcomes
  - This gives you a **95% confidence interval**

### Why This Matters

**Without Monte Carlo:**
- You test ONE parameter set (e.g., n_stocks=5, holding_period=21)
- You get ONE result
- You don't know if it's luck or if the strategy is robust

**With Monte Carlo:**
- You test 100 different parameter combinations
- You see the **distribution** of outcomes
- You know:
  - Is the strategy **robust** (consistent results)?
  - What's the **worst-case** scenario?
  - What's the **expected** performance?
  - How **sensitive** is it to parameter choices?

### Real Example

**Scenario**: You run Monte Carlo with 100 simulations

**Results**:
```
Total Return:
  Mean: 0.4523 (45.23% average)
  95% CI: [0.1234, 0.7891]  (12.34% to 78.91%)
  Range: [0.0123, 0.9123]   (1.23% worst, 91.23% best)
```

**Interpretation**:
- ✅ Strategy is robust: Even worst case (1.23%) is positive
- ✅ 95% of runs fall between 12% and 79% return
- ✅ Mean is 45%, so that's your expected outcome
- ⚠️ Wide range (1% to 91%) suggests some parameter sensitivity

---

## Method 2: Bootstrap Simulation (`bootstrap_simulation`)

### What It Actually Does

This tests: **"What if the market had moved differently?"**

#### Step 1: Resample Historical Data
Instead of using the actual chronological order of returns, it:
- Divides historical returns into **blocks** (e.g., 21-day blocks)
- **Randomly resamples** these blocks with replacement
- Creates a **new, shuffled timeline** of returns

**Example:**
```
Original timeline: [Day1-21, Day22-42, Day43-63, ...]
Bootstrap sample:  [Day43-63, Day1-21, Day43-63, Day22-42, ...]
                  (same blocks, different order)
```

#### Step 2: Run Backtest on Shuffled Data
- Uses the **same strategy parameters**
- But applies them to the **resampled return sequence**
- Tests if strategy works under different market path scenarios

#### Step 3: Repeat 100 Times
Each simulation creates a different shuffled timeline and tests the strategy.

### Why This Matters

**Tests path dependency**: Does your strategy depend on specific market sequences, or would it work in different scenarios?

**Example**: 
- Strategy might work great in 2010-2015 bull market
- But what if that period came later? Or earlier?
- Bootstrap tests this by reordering market conditions

---

## Visual Analogy

### Parameter Monte Carlo
Imagine you're testing a car's fuel efficiency:

**Without Monte Carlo:**
- Test once: Drive 60 mph, get 30 mpg
- Conclusion: "Car gets 30 mpg" (but what if you drove 55 or 65?)

**With Monte Carlo:**
- Test 100 times: Randomly vary speed (50-70 mph), road conditions, etc.
- Results: 30 mpg average, but range is 25-35 mpg
- Conclusion: "Car gets 30 mpg on average, but can vary 25-35 mpg depending on conditions"

### Bootstrap Simulation
Imagine you're testing a recipe:

**Without Bootstrap:**
- Make recipe once with ingredients in order: A, B, C, D
- Result: Tastes good
- Conclusion: "Recipe works"

**With Bootstrap:**
- Make recipe 100 times, randomly reordering steps: C, A, D, B or B, D, A, C, etc.
- Results: Sometimes good, sometimes bad
- Conclusion: "Recipe only works with specific order" (path-dependent) or "Recipe works regardless of order" (robust)

---

## What You Get From Monte Carlo

### 1. Robustness Assessment
- **Low std**: Strategy works consistently → Robust ✅
- **High std**: Strategy very sensitive → Fragile ⚠️

### 2. Confidence Intervals
- "I'm 95% confident my return will be between X% and Y%"
- Better than single point estimate

### 3. Risk Assessment
- Worst-case scenario (5th percentile)
- Best-case scenario (95th percentile)
- Understand tail risks

### 4. Parameter Sensitivity
- Which parameters matter most?
- Should you optimize n_stocks or holding_period?
- Monte Carlo shows which parameters drive performance variance

### 5. Strategy Comparison
- Compare Monte Carlo results across strategies
- Which is more robust?
- Which has better worst-case?

---

## Code Flow Example

```python
# Run 100 simulations
for i in range(100):
    # 1. Randomly pick parameters
    n_stocks = random(3, 8)           # e.g., 5
    holding_period = random(15, 30)   # e.g., 22
    
    # 2. Run full backtest with these parameters
    equity = run_backtest_enhanced(
        n_stocks=5,
        holding_period=22,
        strategy="risk_adjusted_momentum"
    )
    
    # 3. Calculate metrics
    metrics = calculate_metrics(equity)
    # Returns: {total_return: 0.45, sharpe: 1.2, ...}
    
    # 4. Store results
    results.append(metrics)

# After 100 runs, analyze all results
summary = analyze_monte_carlo_results(results)
# Shows: mean, std, percentiles, confidence intervals
```

---

## Key Takeaway

**Monte Carlo simulation = Stress testing your strategy**

Instead of asking: *"Does my strategy work with these exact parameters?"*

You ask: *"Does my strategy work across a wide range of reasonable parameters?"*

If yes → **Robust strategy** (you can be confident)
If no → **Fragile strategy** (might be overfitted to specific parameters)

This is why professional quants use Monte Carlo: it separates **robust strategies** from **lucky backtests**.

