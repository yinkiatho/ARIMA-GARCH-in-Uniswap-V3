# Tokka-Labs Uniswap V3 Strategy

Hedging Impermanent Loss and Optimizing Returns with ARIMA-GARCH and Options

This repository contains code and research materials for exploring a novel strategy to minimize impermanent loss and maximize returns in Uniswap V3.

- main simulation Jupyter Notebook is at src/main.ipynb

## Key Features
### Asset Selection:
WBTC/ETH pair chosen for its high correlation and potential to reduce impermanent loss.

## Modeling:
- Hybrid ARIMA-GARCH model forecasts pool prices and estimates volatilities.
- Walk-forward validation technique for model evaluation.

## Hedging:
- Options-based hedging strategy to mitigate impermanent loss.
- Uses BTC-USDT and ETH-USDT options on Deribit exchange.


## Investment Strategy:
- "Hold On Tight" strategy with one or more boundaries, optimized using Monte Carlo simulations.
- Dynamic boundaries for rebalancing during investment periods.

### Project Structure
- data: Contains historical Uniswap V3 pool data and options data.
- models: Contains code for ARIMA-GARCH modeling and Monte Carlo simulations.


## Roadmap
1. Phase 1: Backtesting and Simulation Implementation (in progress)
2. Phase 2: Hedging Strategy & Investment Strategy Optimization
3. Phase 3: Analytics Dashboard Development

### Streamlit-app
https://crypto-whales.streamlit.app


Disclaimer
This project is for research and educational purposes only. It does not constitute financial advice. Investing in cryptocurrencies involves significant risks. Use any strategies at your own risk.

For more details, please refer to the full report: [link to report]
