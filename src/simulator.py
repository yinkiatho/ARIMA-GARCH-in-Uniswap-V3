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
import defi.defi_tools as dft
import math 
import random
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
from ARIMA_GARCH import ARIMA_GARCH
from Backtester import Backtester
from utils import *
import os


class Simulator():
    
    def __init__(self, Address='0xcbcdf9626bc03e24f779434178a73a0b4bad62ed', network=1):
        
        self.Address = Address
        self.network = network
        self.backtester = Backtester(Address, network)
        #self.arima_garch = ARIMA_GARCH((9, 0, 2), (1,1))
        try: 
            self.model_predictions = pd.read_csv('data/pools_daily_weth_btc_arima_garch.csv', delimiter=';', index_col=0)
            self.btc_price = pd.read_csv('data/btc_hist.csv', delimiter=',', index_col=0)
            self.eth_price = pd.read_csv('data/eth_hist.csv', delimiter=',', index_col=0)
            self.futures_data = pd.read_csv('data/futures_data.csv', delimiter=',', index_col=0, parse_dates=True)
        except:
            self.model_predictions = pd.read_csv('../data/pools_daily_weth_btc_arima_garch.csv', delimiter=';', index_col=0)
            self.btc_price = pd.read_csv('../data/btc_hist.csv', delimiter=',', index_col=0)
            self.eth_price = pd.read_csv('../data/eth_hist.csv', delimiter=',', index_col=0)
            self.futures_data = pd.read_csv('../data/futures_data.csv', delimiter=',', index_col=0, parse_dates=True)
        self.start_date = "2023-05-25"
        self.end_date = "2023-12-24"
        
        # Round to nearest second
        self.btc_price.index = pd.to_datetime(self.btc_price.index).round('1s')
        self.eth_price.index = pd.to_datetime(self.eth_price.index).round('1s')
        
    
    def get_futures_price(self, end_date):
        #print(self.futures_data)
        # Get the futures price for the end date
        return self.futures_data.loc[self.futures_data.index <= end_date]['Spot Price'].iloc[-1]
    
    
    def get_current_btc_price(self, date):
        #print(date)
        #print(self.btc_price)
        date = pd.to_datetime(date)
        return self.btc_price.loc[date, 'Close']
    
    def get_current_eth_price(self, date):
        #print(date)
        #print(self.btc_price)
        date = pd.to_datetime(date)
        return self.eth_price.loc[date, 'Close']
    
    def get_options_payoff(self, end_date, list_options, num_contracts=1):
        
        payoff = 0
        curr_eth_price = self.eth_price.loc[end_date]['Close'].iloc[0]
        curr_btc_price = self.btc_price.loc[end_date]['Close'].iloc[0]
        for option in list_options:
            if option['coin'] == 'ETH':
                curr_price = curr_eth_price
            else:
                curr_price = curr_btc_price
            
            if option['Type'] == 'CALL':
                if curr_price > option['strike_price']:
                    payoff += (curr_price - option['strike_price']) * num_contracts          
            else:
                if curr_price < option['strike_price']:
                    payoff += (option['strike_price'] - curr_price) * num_contracts
                    
        return payoff
    
    
    def get_options_straddle(self, start_date, end_date, close_ratio, num_contracts=1):
        days_to_expiry = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        overnight_apy = 0.0515
        total_options = []
        curr_eth_price = self.eth_price.loc[start_date]['Close'].iloc[0]
        curr_btc_price = self.btc_price.loc[start_date]['Close'].iloc[0]
        num_btc = 1
        num_eth = int(num_btc / close_ratio)
        
        for coin in ['ETH', 'BTC']:
            if coin == 'ETH':
                curr_price = curr_eth_price
            else:
                curr_price = curr_btc_price
            for type in ['CALL', 'PUT']:
                strike_price = curr_price
                option = {
                    'coin': coin,
                    'strike_price': strike_price,
                    'Time to Expiry': days_to_expiry / 365,
                    'staking_apy': overnight_apy,
                    'Type' : type,
                    'Close Price (USD)': curr_price
                }
                #print(option)
                
                if coin == 'ETH':
                    total_options += [option] * num_eth
                    
                else:
                    total_options += [option] * num_btc
                #total_options.append(option)
                
        return total_options
            
            
        
        
    def get_options(self, start_date, end_date, num_contracts=1):
        days_to_expiry = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        overnight_apy = 0.0515
        price_ranges = [-0.4, -0.3, -0.2, 0.2, 0.3, 0.4]
        total_options = []
        curr_eth_price = self.eth_price.loc[start_date]['Close'].iloc[0]
        curr_btc_price = self.btc_price.loc[start_date]['Close'].iloc[0]
        
        # +- 20, 30, 40% of current price for strike price
        for coin in ['ETH', 'BTC']:
            if coin == 'ETH':
                curr_price = curr_eth_price
            else:
                curr_price = curr_btc_price
                
            for price_range in price_ranges:
                if price_range < 0:
                    type = 'PUT'
                else:
                    type = 'CALL'
                    
                strike_price = curr_price * (1 + price_range)
                option = {
                    'coin': coin,
                    'strike_price': strike_price,
                    'Time to Expiry': days_to_expiry / 365,
                    'staking_apy': overnight_apy,
                    'Type' : type,
                    'Close Price (USD)': curr_price
                }
                #print(option)
                total_options.append(option)
                
        return total_options
        
    def divide_windows(self, windows=1):
        test_period = self.model_predictions[(self.model_predictions.index >= self.start_date) & (self.model_predictions.index <= self.end_date)]

        if windows == 1:
            return [test_period]
        else:
            min_window_length = 1  # Minimum length of each window in months
            total_months = len(test_period) // 30
            print(f"Total Months: {total_months}")

            # Calculate the maximum possible windows with minimally one-month length
            max_windows = min(windows, total_months // min_window_length)

            interval = len(test_period) // max_windows
            test_periods = []

            for i in range(max_windows):
                if i == max_windows - 1:
                    test_periods.append(test_period.iloc[i * interval:])
                else:
                    test_periods.append(test_period.iloc[i * interval:(i + 1) * interval])

            return test_periods
        
    
    def simulate(self, windows=1, risk_params=0.95, initial_investment=1000000):
        
        # 1.  Based on number of windows, divide the data into windows, get start and end dates for each window as well
        test_periods = self.divide_windows(windows)
        
        # For each test period, run the backtester and get the returns for that period, include hedging costs
        
        results = {
            'Test Period': [],
            'Start Date': [],
            'End Date': [],
            'Start Price (WBTC)': [],
            'End Price (WBTC)': [],
            'Lower Bound': [],
            'Upper Bound': [],
            'Final Net Liquidity Value': [],
            'Fee Results': [],
            'Fee USD': [],
            'APR Strategy': [],
            'APR Unbounded': [],
            'Cumulative Investment USD': [],
            'Cumulative Investment WBTC': [],
            'Mean Percentage of Active Liquidity': [],
            'Hedging Costs': [],
            'Payoff': [],
            'Impermanent Loss': [],
            'HODL 50-50': []
            #'Total Results': []
        }
        
        curr_initial_investment = initial_investment
        initialprice0, initialprice1 = self.get_current_btc_price(self.start_date), self.get_current_eth_price(self.start_date)
        initalamount0, initalamount1 = curr_initial_investment * 0.5 / initialprice0, curr_initial_investment * 0.5 / initialprice1
        curr_hodl_value = initial_investment
        for test_period in test_periods:
            previous_hodl_value = curr_hodl_value
            # Add initial results
            results['Test Period'].append(test_period)
            results['Start Date'].append(test_period.index[0])
            results['End Date'].append(test_period.index[-1])

            # Initialize Backtest
            start_date, end_date = test_period.index[0], test_period.index[-1]
            print(f"Test Period: {start_date} to {end_date}")
            #print(f"Real Start Price (WBTC): {test_period['Close (WBTC)'].iloc[0]}, Real End Price (WBTC): {test_period['Close (WBTC)'].iloc[-1]}")
            start_price = test_period['Close (WBTC)'].iloc[0]
            #end_price = test_period['Predicted Close (WBTC)'].iloc[-1] # should be futures price
            end_price = self.get_futures_price(end_date)

            
            start_volatility, end_volatility = test_period['Conditional Volatility'].iloc[0], test_period['Conditional Volatility'].iloc[-1]
            lower_bound, upper_bound = generate_bounds(start_volatility, end_volatility, start_price, end_price, confidence=risk_params)
            
            print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
            results['Lower Bound'].append(lower_bound)
            results['Upper Bound'].append(upper_bound)
            results['Start Price (WBTC)'].append(start_price)
            results['End Price (WBTC)'].append(end_price)
            
            # Run Simulations generate fees
            #res1, res2, res3, fee_outputs = self.backtester.backtest(lower_bound, upper_bound, start_date, end_date)
            #results['Fee Results'].append([res1, res2, res3, fee_outputs])

            res1, exit_value, exit_value_usd, hodl_value_lp = self.backtester.uniswap_strategy_backtest(self.Address, lower_bound, upper_bound, 
                                                                                         start_date, end_date, investment_amount_usd=curr_initial_investment)
            
            res1 = self.backtester.add_IL(lower_bound, upper_bound, res1)
            #print(res1)
            results['Impermanent Loss'].append(res1['IL'].tolist())
            chart1, stats = self.backtester.generate_chart1(res1)
            fees_usd, apr, apr_base, final_net_liquidity, active_liquidity = stats
            
            
            # Inititalize Hedging Costs
            list_options = self.get_options(start_date, end_date)
            #list_options = self.get_options_straddle(start_date, end_date, close_ratio=start_price)
            num_contracts = (initial_investment / 1000000) 
            #print(list_options)
            #total_premiums = 0
            payoff = self.get_options_payoff(end_date, list_options, num_contracts=num_contracts)
            #total_premiums = sum[get_premiums(****) for option in options] * 1.0003 * num_contracts
            total_premiums = sum([optionPrice(option['coin'], option['strike_price'], option['Time to Expiry'], option['Type'], option['Close Price (USD)'], option['staking_apy']) for option in list_options]) * 1.0003 * num_contracts
            print(f"Total Premiums: {total_premiums}")
            
            
            
            # Calculating HODL Value
            btc_usd_start, btc_usd_end = self.get_current_btc_price(start_date), self.get_current_btc_price(end_date)
            eth_usd_start, eth_usd_end = self.get_current_eth_price(start_date), self.get_current_eth_price(end_date)
            curr_hodl_value = generate_hodl([btc_usd_start, btc_usd_end, eth_usd_start, eth_usd_end], [initalamount0, initalamount1])
            
            final_il = res1['IL'].iloc[-1]
            final_lp_value = (1 + final_il) * hodl_value_lp
            #curr_hodl_value = hodl_final_value
            #print(f"Final LP Value using IL: {final_lp_value}")
            
            # Add Gas Fees
            
            # Add Hedging Costs
            #curr_initial_investment = exit_value_usd + fees_usd - total_premiums + payoff 
            curr_initial_investment = final_lp_value + fees_usd - total_premiums + payoff
            
            
            
            
            # Using IL as comparison to calculate final LP value
            
            print(f"Fees: {fees_usd}, Hedging Costs: {total_premiums}, Payoff: {payoff}")
            print(f'Current Value USD at end of period {start_date} to {end_date}: {curr_initial_investment} USD')
            print(f"HODL 50-50- Start Value: {previous_hodl_value}, End_Value: {curr_hodl_value}")
            print("-------------------------------------------------------------------")
            
                    
            # Exit LP add results
            results['Fee USD'].append(fees_usd)
            results['APR Strategy'].append(apr)
            results['APR Unbounded'].append(apr_base)
            results['Final Net Liquidity Value'].append(final_net_liquidity)
            results['Mean Percentage of Active Liquidity'].append(active_liquidity)
            results['Fee Results'].append(chart1)
            results['Hedging Costs'].append(total_premiums)
            results['Payoff'].append(payoff)
            results['Cumulative Investment USD'].append(curr_initial_investment)
            results['Cumulative Investment WBTC'].append(exit_value)
            results['HODL 50-50'].append(curr_hodl_value)
            
        
        
        # 2.  For each window, run the backtester to get the returns for that window
        
        
        # 3.  Metrics for the window, Hedging Costs + Fees Earned + Any possible transaction costs
        
        return results
    
    
if __name__ == '__main__':
    
    sim = Simulator()
    results = sim.simulate(windows=1, risk_params=0.55)
    