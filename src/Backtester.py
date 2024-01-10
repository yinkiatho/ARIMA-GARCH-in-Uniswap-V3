import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
import sys
from utils import *
import os

class Backtester():
    
    
    def __init__(self, Address='0xcbcdf9626bc03e24f779434178a73a0b4bad62ed', network=1):
        print(os.getcwd())
        print("Initializing Backtester...")
        self.Address = Address
        self.network = network
        
        try:   
            self.btc_price = pd.read_csv('data/btc_hist.csv', delimiter=',', index_col=0)
            self.eth_price = pd.read_csv('data/eth_hist.csv', delimiter=',', index_col=0)
        except:
            self.btc_price = pd.read_csv('../data/btc_hist.csv', delimiter=',', index_col=0)
            self.eth_price = pd.read_csv('../data/eth_hist.csv', delimiter=',', index_col=0)
        
        # Round to nearest second
        self.btc_price.index = pd.to_datetime(self.btc_price.index).round('1s')
        self.eth_price.index = pd.to_datetime(self.eth_price.index).round('1s')
        
        
    def get_current_btc_price(self, date):
        date = pd.to_datetime(date).round('1s')
        return self.btc_price.loc[date, 'Close']
    
    def get_current_eth_price(self, date):
        date = pd.to_datetime(date).round('1s')
        return self.eth_price.loc[date, 'Close']
        
    
    
    def add_IL(self, mini, maxi, dpd):
        ## CALCULATING IL
        p0 = dpd['close'].iloc[0]
        dpd['k'] = dpd['close'].astype(float)/p0
        
        k0 = dpd.loc[dpd['k']>maxi/dpd['close'],'k'].astype(float)
        dpd.loc[dpd['k']>maxi/dpd['close'],'IL'] = ((maxi/p0)**0.5 - k0*(1-(p0/maxi)**0.5) -1) / (k0*(1-(p0/maxi)**0.5) + 1 - (mini/p0)**0.5)

        k1 = dpd.loc[(dpd['k']<=maxi/dpd['close']) & (dpd['k']>=mini/dpd['close']),'k'].astype(float)
        dpd.loc[(dpd['k']<=maxi/dpd['close']) & (dpd['k']>=mini/dpd['close']),'IL'] = (2*(k1**0.5) - 1 - k1) / (1 + k1 - (mini/p0)**0.5 - k1*(p0/maxi))
        
        k2 = dpd.loc[dpd['k']<mini/dpd['close'],'k'].astype(float)
        dpd.loc[dpd['k']<mini/dpd['close'],'IL'] = ((k2*((p0/mini)**0.5-1) -1 + (mini/p0)**0.5)) / (k2*(1-(p0/maxi)**0.5) + 1 - (mini/p0))
        
        return dpd
    

    
    def generate_hodl(self, prices, initial_investment):
        btc_start, btc_end, eth_start, eth_end = prices
        
        # Calculate the initial investment for each coin (50-50 split)
        btc_investment = initial_investment / 2
        eth_investment = initial_investment / 2
        
        # Calculate the final value for each coin based on the price change
        btc_final_value = btc_investment * (btc_end / btc_start)
        eth_final_value = eth_investment * (eth_end / eth_start)
        
        # Calculate the total final value in USD
        total_final_value = btc_final_value + eth_final_value
        
        return total_final_value
    
    
    
    def uniswap_strategy_backtest(self, pool, min_range, max_range, start_date, end_date, btc_usd=44109, investment_amount_usd=1000000, base = 0):
        
        opt = {"days": 30, "protocol": 0, "priceToken": 0, "period": "daily"}
    
        # Timestamp Handling
        start_timestamp, end_timestamp = convert_to_unix(start_date), convert_to_unix(end_date)

        # Fetching Hourly Price Data
        hourly_price_data = graphTwo(1, pool, start_timestamp, end_timestamp)
        
        btc_usd_start, btc_usd_end = self.get_current_btc_price(convert_unix_to_datetime(start_timestamp)), self.get_current_btc_price(convert_unix_to_datetime(end_timestamp))
        eth_usd_start, eth_usd_end = self.get_current_eth_price(convert_unix_to_datetime(start_timestamp)), self.get_current_eth_price(convert_unix_to_datetime(end_timestamp))
        hodl_final_value = self.generate_hodl([btc_usd_start, btc_usd_end, eth_usd_start, eth_usd_end], investment_amount_usd)
        
        print(f"Initial Investment USD: {investment_amount_usd}, BTC-USD: {btc_usd_start}")
        investment_amount = investment_amount_usd/btc_usd_start
        
        if len(hourly_price_data) > 0:
            backtest_data = hourly_price_data.iloc[::-1].reset_index(drop=True)
            print(f"Backtest Data: {backtest_data}")
            decimal0 =  backtest_data.iloc[0]['pool.token0.decimals']
            decimal1 = backtest_data.iloc[0]['pool.token1.decimals']
            entry_price = 1 / backtest_data['close'].iloc[-1]  if opt["priceToken"] == 1 else backtest_data['close'].iloc[-1] 
            token_decimals_diff = decimal1 - decimal0
        
            tokens = tokens_for_strategy(min_range, max_range, investment_amount, entry_price, token_decimals_diff)
        
            liquidity = liquidity_for_strategy(
                entry_price, min_range, max_range, tokens[0], tokens[1],
                decimal0=decimal0, decimal1=decimal1
            )
        
            unb_liquidity = liquidity_for_strategy(
                entry_price, pow(1.0001, -887220), pow(1.0001, 887220),
                tokens[0], tokens[1], decimal0=decimal0, decimal1=decimal1
            )
        
            hourly_backtest = calc_fees(backtest_data, decimal0, decimal1, opt["priceToken"], liquidity, unb_liquidity, investment_amount, min_range, max_range)
            
            final_result = pivot_fee_data(hourly_backtest, opt["priceToken"], investment_amount) if opt["period"] == "daily" else hourly_backtest
            
            # Rough estimation not used
            total_exit_value = self.get_exit_value(hourly_price_data, min_range=min_range, max_range=max_range,
                                                   tick_spacing=60, initial_amount0=investment_amount/2)
            
            return final_result, total_exit_value, hodl_final_value
        
        
    def generate_chart1(self, dpd):
        return chart1(dpd)
    
    def get_exit_value(self, df, initial_amount0, min_range, max_range, tick_spacing=60):
        
        initial_amount1 = initial_amount0 * (1/ df['close'].iloc[-1])
        print(f"Initial Amounts: {initial_amount0} token0, {initial_amount1} token1")
        initial_price = df['close'].iloc[-1]
        final_price = df['close'].iloc[0]
        
        if final_price > initial_price and final_price < max_range:
            ratio = (final_price - initial_price) / (max_range - initial_price)
            final_amount1 = initial_amount1 * (1 - ratio)
            final_amount0 = initial_amount0 * (1 + ratio)
            
        elif final_price < initial_price and final_price > min_range:
            ratio = (initial_price - final_price) / (initial_price - min_range)
            final_amount1 = initial_amount1 * (1 + ratio)
            final_amount0 = initial_amount0 * (1 - ratio)
            
        elif final_price > max_range:
            final_amount0 = initial_amount0 * 2
            final_amount1 = 0
            
        elif final_price < min_range:
            final_amount0 = 0
            final_amount1 = initial_amount1 * 2
            
        #print(f"Final Amounts: {final_amount0} token0, {final_amount1} token1")
        total_exit_value = final_amount0 + final_amount1 * final_price
        return total_exit_value
        


        
if __name__ == "__main__":

    backtester = Backtester()
    Adress = "0xcbcdf9626bc03e24f779434178a73a0b4bad62ed"
    res, total_exit_value, exit_value_usd = backtester.uniswap_strategy_backtest(Adress, 0.04177481929059751, 0.07653292116574624, start_date="2023-05-25", end_date="2023-12-24")
    chart1 = chart1(res)
    print(res)
    
    
            
        
        