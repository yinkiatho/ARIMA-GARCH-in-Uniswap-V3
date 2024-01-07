import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model   
from sklearn import metrics 
import defi.defi_tools as dft
import math 
import random
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
from ARIMA_GARCH import ARIMA_GARCH
from Backtester import Backtester
from utils import *

class Simulator():
    
    def __init__(self, Address='0xcbcdf9626bc03e24f779434178a73a0b4bad62ed', network=1):
        
        self.Address = Address
        self.network = network
        self.backtester = Backtester(Address, network)
        self.arima_garch = ARIMA_GARCH((1,1,1), (1,1))
        self.model_predictions = pd.read_csv('data/pools_daily_weth_btc_arima_garch.csv', delimiter=';', index_col=0)
        self.start_date = "2023-05-25"
        self.end_date = "2023-12-25"
        
    def divide_windows(self, windows=1):
        
        test_period = self.model_predictions[(self.model_predictions.index >= self.start_date) & (self.model_predictions.index <= self.end_date)]
        
        if windows == 1:
            return [test_period]
        
        else:
            interval = int(len(test_period) // windows)
            test_periods = []
            for i in range(windows):
                
                if i == windows - 1:
                    test_periods.append(test_period.iloc[i*interval:])
                else:
                    test_periods.append(test_period.iloc[i*interval:(i+1)*interval])
        
        return test_periods
        
    
    def simulate(self, windows=1, risk_params=0.95, initial_investment=1000000):
        
        # 1.  Based on number of windows, divide the data into windows, get start and end dates for each window as well
        test_periods = self.divide_windows(windows)
        
        # For each test period, run the backtester and get the returns for that period, include hedging costs
        
        results = {
            'Test Period': [],
            'Start Date': [],
            'End Date': [],
            'Final Net Liquidity Value': [],
            'Fee Results': [],
            'Fee USD': [],
            'APR Strategy': [],
            'APR Unbounded': [],
            'Cumulative Investment USD': [],
            'Cumulative Investment WBTC': [],
            'Mean Percentage of Active Liquidity': [],
            'Hedging Costs': [],
            'Total Results': []
        }
        
        curr_initial_investment = initial_investment
        for test_period in test_periods:
            # Add initial results
            results['Test Period'].append(test_period)
            results['Start Date'].append(test_period.index[0])
            results['End Date'].append(test_period.index[-1])

            # Initialize Backtest
            start_date, end_date = test_period.index[0], test_period.index[-1]
            print(f"Test Period: {start_date} to {end_date}")
            start_price = test_period['Predicted Close (WBTC)'].iloc[0]
            end_price = test_period['Predicted Close (WBTC)'].iloc[-1] # should be futures price

            
            start_volatility, end_volatility = test_period['Conditional Volatility'].iloc[0], test_period['Conditional Volatility'].iloc[-1]
            lower_bound, upper_bound = generate_bounds(start_volatility, end_volatility, start_price, end_price, confidence=risk_params)
            
            #lower_bound, upper_bound = 0.04177481929059751, 0.07653292116574624
            print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
            
            # Run Simulations generate fees
            #res1, res2, res3, fee_outputs = self.backtester.backtest(lower_bound, upper_bound, start_date, end_date)
            #results['Fee Results'].append([res1, res2, res3, fee_outputs])

            
            Address =  "0xcbcdf9626bc03e24f779434178a73a0b4bad62ed"
            res1, exit_value, exit_value_usd = self.backtester.uniswap_strategy_backtest(Address, lower_bound, upper_bound, 
                                                                                         start_date, end_date, investment_amount_usd=curr_initial_investment)
            chart1, stats = self.backtester.generate_chart1(res1)
            fees_usd, apr, apr_base, final_net_liquidity, active_liquidity = stats
            
            
            # Inititalize Hedging Costs
            
        
             
            # Add results
            results['Fee USD'].append(fees_usd)
            results['APR Strategy'].append(apr)
            results['APR Unbounded'].append(apr_base)
            results['Final Net Liquidity Value'].append(final_net_liquidity)
            results['Mean Percentage of Active Liquidity'].append(active_liquidity)
            results['Fee Results'].append(chart1)
            
            
            # Add Hedging Costs
            hedgingcosts = 0
            curr_initial_investment = exit_value_usd + fees_usd - hedgingcosts
            
            results['Hedging Costs'].append(hedgingcosts)
            results['Cumulative Investment USD'].append(curr_initial_investment)
            results['Cumulative Investment WBTC'].append(exit_value)
            
            print(f"Exit WBTC: {exit_value}, Exit USD: {exit_value_usd}, Fees: {fees_usd}, Hedging Costs: {hedgingcosts}")
            print(f'Current Value USD at end of period {start_date} to {end_date}: {curr_initial_investment} USD')
            print("-------------------------------------------------------------------")
                    
            # Exit LP
        
        
        # 2.  For each window, run the backtester to get the returns for that window
        
        # 3.  Metrics for the window, Hedging Costs + Fees Earned + Any possible transaction costs
        
        return results
    
    
if __name__ == '__main__':
    
    sim = Simulator()
    sim.simulate(windows=10, risk_params=0.90)