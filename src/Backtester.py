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
        self.btc_price = pd.read_csv('data/btc_hist.csv', delimiter=',', index_col=0)
        
        # Round to nearest second
        self.btc_price.index = pd.to_datetime(self.btc_price.index).round('1s')
        
        
    def get_current_btc_price(self, date):
        #print(date)
        #print(self.btc_price)
        date = pd.to_datetime(date)
        return self.btc_price.loc[date, 'Close']
        
    ## FUNCTION THAT DECIDES AMOUNT OF EACH TOKEN FROM INITIAL CAPITAL AND PRICE RANGE
    def setPosition(self, capital, pmin, pmax, r50 = None, dpd = None):
        
        ## GETTING DEFAULT DATAFRAME IF NOT PROVIDED
        if dpd == None:
            dpd = self.dpd

        ## GETTING LATEST PRICE
        p = dpd['close'].iloc[-1] 

        ## SETTING PRICE RANGE FOR 50/50 PORTFOLIO 
        if r50 != None:
            pmin = p / r50
            pmax = p * r50
        
        ## CALCULATING AMOUNT OF TOKEN 0/1
        decimal0=dpd.iloc[0]['pool.token0.decimals']
        decimal1=dpd.iloc[0]['pool.token1.decimals']
        decimal=decimal1-decimal0
        
        SMIN=np.sqrt(pmin* 10 ** (decimal))   
        SMAX=np.sqrt(pmax* 10 ** (decimal))  
        
        target = p
        sqrt0 = np.sqrt(p* 10 ** (decimal))
        dpd['price0'] = dpd['close']

        if sqrt0>SMIN and sqrt0<SMAX:
                deltaL = target / ((sqrt0 - SMIN)  + (((1 / sqrt0) - (1 / SMAX)) * (dpd['price0'].iloc[-1]* 10 ** (decimal))))
                amount1 = deltaL * (sqrt0-SMIN)
                amount0 = deltaL * ((1/sqrt0)-(1/SMAX))* 10 ** (decimal)
        
        elif sqrt0<SMIN:
                deltaL = target / (((1 / SMIN) - (1 / SMAX)) * (dpd['price0'].iloc[-1]))
                amount1 = 0
                amount0 = deltaL * (( 1/SMIN ) - ( 1/SMAX ))

        else:
                deltaL = target / (SMAX-SMIN) 
                amount1 = deltaL * (SMAX-SMIN)
                amount0 = 0
        
        id0,id1 = getTokenId(self.Address)
        token0PriceUSD = getTokenPrices(id0)
        token1PriceUSD = getTokenPrices(id1)

        print("Amounts Ratio: ",amount0,amount1)
        multiplier = capital / (token0PriceUSD*amount0 + token1PriceUSD*amount1)

        amount0 *= multiplier
        amount1 *= multiplier
        print("Amounts to Deposit: ",amount0,amount1)

        ## GETTING LIQUIDITY BASED ON MY AMOUNT AND POOL AMOUNT
        myliquidity = get_liquidity(dpd['price0'].iloc[-1],pmin,pmax,amount0,amount1,decimal0,decimal1)
        totalLiquidity = get_liquidity(dpd['price0'].iloc[-1],pmin,pmax,dpd['pool.totalValueLockedToken0'][0],dpd['pool.totalValueLockedToken1'][0],decimal0,decimal1)
        #poolLiquidity = dpd['pool.liquidity'].mean() # REMOVED COZ RANGE NOT SPECIFIED
        self.myliquidity = myliquidity

        print("myLiquidity: ",myliquidity)
        print("totalLiquidity: ",totalLiquidity)
        liquidityPerc = myliquidity/totalLiquidity
        print("Liquidity%: ", liquidityPerc)
        
        return (amount0, amount1, liquidityPerc)
        
    def checkTransactionGasFees(amountBefore0, amountBefore1, amountAfter0, amountAfter1):
        if (amountAfter0 - amountBefore0) > 0 and (amountAfter1 - amountBefore1) > 0:
             return getGasPrice()
        elif (amountAfter0 - amountBefore0) < 0 and (amountAfter1 - amountBefore1) < 0:
             return getGasPrice()
        else:
             return 2 * getGasPrice()
    
    def backtest(self, mini, maxi, startdate, enddate=None, base=0, initial_investment=1000000, curr_price_eth = 2286.71):
        
        
        startfrom, enddate = convert_to_unix(startdate), convert_to_unix(enddate)
        #print(startfrom, enddate)
        dpd = graphTwo(self.network, self.Address, startfrom, enddate)

        ## CALCULATING LIQUIDITY PERCENTAGE
        amt0, amt1, liquidityPerc = self.setPosition(initial_investment, mini, maxi)

        ## FINAL FEE CALCULATION
        dpd['myfee0'] = dpd['fee0token'] * liquidityPerc
        dpd['myfee1'] = dpd['fee1token'] * liquidityPerc

        '''
        dpd['myfee0'] = dpd['fee0token'] * myliquidity * dpd['ActiveLiq'] / 100
        dpd['myfee1'] = dpd['fee1token'] * myliquidity * dpd['ActiveLiq'] / 100
        '''
        
        ## CALCULATING IL
        p0 = dpd['close'].iloc[0]
        dpd['k'] = dpd['close'].astype(float)/p0
        
        k0 = dpd.loc[dpd['k']>maxi/dpd['close'],'k'].astype(float)
        dpd.loc[dpd['k']>maxi/dpd['close'],'IL'] = ((maxi/p0)**0.5 - k0*(1-(p0/maxi)**0.5) -1) / (k0*(1-(p0/maxi)**0.5) + 1 - (mini/p0)**0.5)

        k1 = dpd.loc[(dpd['k']<=maxi/dpd['close']) & (dpd['k']>=mini/dpd['close']),'k'].astype(float)
        dpd.loc[(dpd['k']<=maxi/dpd['close']) & (dpd['k']>=mini/dpd['close']),'IL'] = (2*(k1**0.5) - 1 - k1) / (1 + k1 - (mini/p0)**0.5 - k1*(p0/maxi))
        
        k2 = dpd.loc[dpd['k']<mini/dpd['close'],'k'].astype(float)
        dpd.loc[dpd['k']<mini/dpd['close'],'IL'] = ((k2*((p0/mini)**0.5-1) -1 + (mini/p0)**0.5)) / (k2*(1-(p0/maxi)**0.5) + 1 - (mini/p0))

        print(dpd)

        #print(dpd)
        dpd, myliquidity, initial_deposit = self.initial_transform(mini, maxi, startdate, enddate, dpd, base, initial_investment, curr_price_eth)
    
        #print(dpd)
        final1, final2, final3, results = chart1(dpd,base,myliquidity, initial_amounts=initial_deposit)
        
        return final1, final2, final3, results
        
    
    def initial_transform(self, mini, maxi, startdate, enddate, dpd, base=0, initial_inv=1000000, curr_price_eth = 2286.71):
        #startfrom, enddate = convert_to_unix(startdate), convert_to_unix(enddate)
        #print(dpd)
        dpd['date']=dpd['periodStartUnix'].apply(convert_unix_to_datetime)  
            
        decimal0=dpd.iloc[0]['pool.token0.decimals']
        decimal1=dpd.iloc[0]['pool.token1.decimals']
        decimal=decimal1-decimal0
        dpd['fg0']=((dpd['feeGrowthGlobal0X128'])/(2**128))/(10**decimal0)
        dpd['fg1']=((dpd['feeGrowthGlobal1X128'])/(2**128))/(10**decimal1)
        
        
        num_eth = (initial_inv/2)/curr_price_eth

        target = dpd['close'].iloc[-1] 
        base = 0
        initial_deposit = [num_eth * target, num_eth]   # Amount of WBTC, WETH to deposit initially
        initial_investment0 = initial_deposit[0] * 2
        #Calculate F0G and F1G (fee earned by an unbounded unit of liquidity in one period)
        dpd['fg0shift']=dpd['fg0'].shift(-1)
        dpd['fg1shift']=dpd['fg1'].shift(-1)
        dpd['fee0token']=dpd['fg0']-dpd['fg0shift'] 
        dpd['fee1token']=dpd['fg1']-dpd['fg1shift']

        # calculate my liquidity
        SMIN=np.sqrt(mini* 10 ** (decimal))   
        SMAX=np.sqrt(maxi* 10 ** (decimal))  

        if base == 0:
            sqrt0 = np.sqrt(dpd['close'].iloc[-1]* 10 ** (decimal))
            dpd['price0'] = dpd['close']

        else:
            sqrt0= np.sqrt(1/dpd['close'].iloc[-1]* 10 ** (decimal))
            dpd['price0']= 1/dpd['close']
    
        if sqrt0>SMIN and sqrt0<SMAX:
                deltaL = initial_investment0 / ((sqrt0 - SMIN)  + (((1 / sqrt0) - (1 / SMAX)) * (dpd['price0'].iloc[-1]* 10 ** (decimal))))
                amount1 = deltaL * (sqrt0-SMIN)
                amount0 = deltaL * ((1/sqrt0)-(1/SMAX))* 10 ** (decimal)
        
        elif sqrt0<SMIN:
                deltaL = initial_investment0 / (((1 / SMIN) - (1 / SMAX)) * (dpd['price0'].iloc[-1]))
                amount1 = 0
                amount0 = deltaL * (( 1/SMIN ) - ( 1/SMAX ))

        else:
                deltaL = initial_investment0 / (SMAX-SMIN) 
                amount1 = deltaL * (SMAX-SMIN)
                amount0 = 0


        print("Amounts:",amount0,amount1)

        #print(dpd['price0'].iloc[-1],mini,maxi)
        #print((dpd['price0'].iloc[-1],mini,maxi,amount0,amount1,decimal0,decimal1))
        myliquidity = get_liquidity(dpd['price0'].iloc[-1],mini,maxi,amount0,amount1,decimal0,decimal1)
        unboundedliquidity = get_liquidity(dpd['price0'].iloc[-1],1.0001**(-887220),1.0001**887220, amount0, amount1,decimal0,decimal1)


        print("OK myliquidity",myliquidity)

        # Calculate ActiveLiq
        dpd['ActiveLiq'] = 0
        dpd['amount0'] = 0
        dpd['amount1'] = 0
        dpd['amount0unb'] = 0
        dpd['amount1unb'] = 0

        if base == 0:

            for i, row in dpd.iterrows():
                if dpd['high'].iloc[i]>mini and dpd['low'].iloc[i]<maxi:
                    dpd.iloc[i,dpd.columns.get_loc('ActiveLiq')] = (min(maxi,dpd['high'].iloc[i]) - max(dpd['low'].iloc[i],mini)) / (dpd['high'].iloc[i]-dpd['low'].iloc[i]) * 100
                else:
                    dpd.iloc[i,dpd.columns.get_loc('ActiveLiq')] = 0
       
                amounts= get_amounts(dpd['price0'].iloc[i],mini,maxi,myliquidity,decimal0,decimal1)
                dpd.iloc[i,dpd.columns.get_loc('amount0')] = amounts[1]
                dpd.iloc[i,dpd.columns.get_loc('amount1')]  = amounts[0]
        
                amountsunb= get_amounts((dpd['price0'].iloc[i]),1.0001**(-887220),1.0001**887220,unboundedliquidity,decimal0,decimal1)
                dpd.iloc[i,dpd.columns.get_loc('amount0unb')] = amountsunb[1]
                dpd.iloc[i,dpd.columns.get_loc('amount1unb')] = amountsunb[0]


        else:

            for i, row in dpd.iterrows():

                if (1/ dpd['low'].iloc[i])>mini and (1/dpd['high'].iloc[i])<maxi:
                    dpd.iloc[i,dpd.columns.get_loc('ActiveLiq')] = (min(maxi,1/dpd['low'].iloc[i]) - max(1/dpd['high'].iloc[i],mini)) / ((1/dpd['low'].iloc[i])-(1/dpd['high'].iloc[i])) * 100
                else:
                    dpd.iloc[i,dpd.columns.get_loc('ActiveLiq')] = 0

                amounts= get_amounts((dpd['price0'].iloc[i]*10**(decimal)),mini,maxi,myliquidity,decimal0,decimal1)
                dpd.iloc[i,dpd.columns.get_loc('amount0')] = amounts[0]
                dpd.iloc[i,dpd.columns.get_loc('amount1')] = amounts[1]

                amountsunb= get_amounts((dpd['price0'].iloc[i]),1.0001**(-887220),1.0001**887220,1,decimal0,decimal1)
                dpd.iloc[i,dpd.columns.get_loc('amount0unb')] = amountsunb[0]
                dpd.iloc[i,dpd.columns.get_loc('amount1unb')] = amountsunb[1]

        ## Final fee calculation
        dpd['myfee0'] = dpd['fee0token'] * myliquidity * dpd['ActiveLiq'] / 100
        dpd['myfee1'] = dpd['fee1token'] * myliquidity * dpd['ActiveLiq'] / 100
         
        return dpd, myliquidity, initial_deposit
    
    
    
    def uniswap_strategy_backtest(self, pool, min_range, max_range, start_date, end_date, btc_usd=44109, investment_amount_usd=1000000, base = 0):
        
        opt = {"days": 30, "protocol": 0, "priceToken": 0, "period": "daily"}
    
        # Timestamp Handling
        start_timestamp, end_timestamp = convert_to_unix(start_date), convert_to_unix(end_date)

        # Fetching Hourly Price Data
        hourly_price_data = graphTwo(1, pool, start_timestamp, end_timestamp)
        print(hourly_price_data)
        
        btc_usd_start = self.get_current_btc_price(convert_unix_to_datetime(start_timestamp))
        btc_usd_end = self.get_current_btc_price(convert_unix_to_datetime(end_timestamp))
        
        print(f"BTC-USD Start: {btc_usd_start}, BTC-USD End: {btc_usd_end}")
        
        print(f"Initial Investment USD: {investment_amount_usd}, BTC-USD: {btc_usd_start}")
    
        investment_amount = investment_amount_usd/btc_usd_start
        print(f"Initial Investment: {investment_amount} WBTC")
        

        if len(hourly_price_data) > 0:
            backtest_data = hourly_price_data.iloc[::-1].reset_index(drop=True)
            #print(backtest_data)
            decimal0 = backtest_data.iloc[0]['pool.token0.decimals']
            decimal1  =backtest_data.iloc[0]['pool.token1.decimals']
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
            total_exit_value = self.get_exit_value(hourly_price_data, min_range=min_range, max_range=max_range,
                                                   tick_spacing=60, initial_amount0=investment_amount/2)
            
            #print(f"Total Exit Value: {total_exit_value}")
            exit_value_usd = total_exit_value * btc_usd_end
            

            return final_result, total_exit_value, exit_value_usd
        
        
    def generate_chart1(self, dpd):
        return chart1(dpd)
    
    def get_exit_value(self, df, initial_amount0, min_range, max_range, tick_spacing=60):
        #print(df)
        
        initial_amount1 = initial_amount0 * (1/ df['close'].iloc[-1])
        print(f"Initial Amounts: {initial_amount0} token0, {initial_amount1} token1")
        initial_price = df['close'].iloc[-1]
        final_price = df['close'].iloc[0]
        
        if final_price > initial_price and final_price < max_range:
            ratio = (final_price - initial_price) / (max_range - initial_price)
            final_amount1 = initial_amount1 * (1 + ratio)
            final_amount0 = initial_amount0 * (1 - ratio)
            
        elif final_price < initial_price and final_price > min_range:
            ratio = (initial_price - final_price) / (initial_price - min_range)
            final_amount1 = initial_amount1 * (1 - ratio)
            final_amount0 = initial_amount0 * ( 1 + ratio)
            
        elif final_price > max_range:
            final_amount0 = 0
            final_amount1 = initial_amount1 * 2
            
        elif final_price < min_range:
            final_amount0 = initial_amount0 * 2
            final_amount1 = 0
            
        print(f"Final Amounts: {final_amount0} token0, {final_amount1} token1")
        
        total_exit_value = final_amount0 + final_amount1 * final_price
        
        return total_exit_value
        
    

        
        
        

        
if __name__ == "__main__":

    backtester = Backtester()
    Adress = "0xcbcdf9626bc03e24f779434178a73a0b4bad62ed"
    res, total_exit_value, exit_value_usd = backtester.uniswap_strategy_backtest(Adress, 0.04177481929059751, 0.07653292116574624, start_date="2023-05-25", end_date="2023-12-24")
    chart1 = chart1(res)
    print(res)
    
    
            
        
        