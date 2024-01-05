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
                
                test_periods.append(test_period.iloc[i*interval:(i+1)*interval])
        
        return test_periods
        
    
    def simulate(self, windows=1, risk_params=0.95):
        
        # 1.  Based on number of windows, divide the data into windows, get start and end dates for each window as well
        test_periods = self.divide_windows(windows)
        
        # For each test period, run the backtester and get the returns for that period, include hedging costs
        
        results = {
            'Test Period': [],
            'Start Date': [],
            'End Date': [],
            'Fee Results': [],
            'Hedging Costs': [],
            'Total Results': []
        }
        for test_period in test_periods:
            start_date, end_date = test_period.index[0], test_period.index[-1]
            start_price = test_period['Predicted Close (WBTC)'].iloc[0]
            end_price = test_period['Predicted Close (WBTC)'].iloc[-1] # should be futures price
            
            start_volatility, end_volatility = test_period['Conditional Volatility'].iloc[0], test_period['Conditional Volatility'].iloc[-1]
            lower_bound, upper_bound = generate_bounds(start_price, start_volatility, risk_params)
            
            # Inititalize Hedging Costs
            
            # Add Hedging Costs
            
            # Run Simulations generate fees
            res1, res2, res3 = self.backtester.backtest(lower_bound, upper_bound, start_date, end_date)
            results['Fee Results'].append([res1, res2, res3])
            print(res1)
            print(res2)
            print(res3)
                    
            # Exit LP
        
        
        # 2.  For each window, run the backtester to get the returns for that window
        
        # 3.  Metrics for the window, Hedging Costs + Fees Earned + Any possible transaction costs
        
        
        
        
        
        return
    
    
if __name__ == '__main__':
    
    sim = Simulator()
    sim.simulate()