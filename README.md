# Improving Basket Trading Using Bayesian Optimization

## Overview
Traditional econometric approaches such as the Johansen cointegration test often fail to generalize in real financial markets due to regime shifts, structural breaks, and non-stationary dynamics.

This project demonstrates how **Bayesian Optimization**, a global and sample-efficient optimization technique, can be used to **improve cointegration-inspired basket trading performance** by directly optimizing out-of-sample trading metrics instead of relying on strict statistical assumptions.

---

## Key Contributions
- Empirical demonstration of Johansen cointegration failure on US tech stocks
- Reframing basket construction as a black-box optimization problem
- Bayesian Optimization of basket weights under fixed trading rules
- Performance-driven approach that generalizes beyond classical theory

---

## Dataset
- Assets: **AAPL, MSFT, GOOGL**
- Source: Yahoo Finance
- Frequency: Daily Adjusted Close prices
- Period: 2017–Present

---

## Methodology

### Phase 1 — Data Preparation
- Data sourced from Yahoo Finance
- Adjusted Close prices used
- Log-price transformation applied

### Phase 2 — Statistical Validation
- Augmented Dickey–Fuller tests confirm I(1) behavior
- First-differenced log prices are stationary

### Phase 3 — Classical Baseline (Johansen Test)
- Lag selection using VAR + BIC
- Johansen trace test applied
- Result: **No statistically significant cointegration**
- Conclusion: classical approach fails to generalize

### Phase 4 — Trading Formulation
- Cointegration-inspired mean-reversion strategy
- Z-score based entry/exit rules
- Fixed trading logic to avoid optimization bias

### Phase 5 — Bayesian Optimization
- Basket weights treated as decision variables
- Objective: maximize out-of-sample Sharpe ratio
- Bayesian Optimization used for global search
- Gaussian Process surrogate + Expected Improvement acquisition

---

## Trading Strategy
- Spread: linear combination of log prices
- Signal: rolling Z-score
- Entry:
  - Long if Z < -2
  - Short if Z > +2
- Exit: |Z| < 0.5
- Transaction costs included

---

## Results
- Bayesian Optimization discovers weight combinations
  that outperform classical econometric baselines
- Improved Sharpe ratio and stability
- Demonstrates superiority of performance-driven optimization

---

## How to Run

```bash
git clone https://github.com/yourusername/bayesian-optimization-basket-trading.git
cd bayesian-optimization-basket-trading
pip install -r requirements.txt
