from datetime import datetime, timedelta
from GraphBacktest import *
from numberstwo import *
from utils import *
import pandas as pd

def date_by_days_ago(days, end_date=None):
    date = datetime.utcfromtimestamp(end_date) if end_date is not None else datetime.utcnow()
    result_date = date - timedelta(days=days)
    return int(result_date.timestamp())


# Assuming the existence of some functions like poolById, getPoolHourData, DateByDaysAgo, tokensForStrategy,
# liquidityForStrategy, calcFees, pivotFeeData. You need to define these functions accordingly.

def uniswap_strategy_backtest(pool, min_range, max_range, start_date, end_date, btc_usd=44109, investment_amount_usd=1000000, base = 0):
    
    investment_amount = investment_amount_usd/btc_usd
    print(f"Initial Investment: {investment_amount} WBTC")
    
    opt = {"days": 30, "protocol": 0, "priceToken": 0, "period": "daily"}
    
    # Timestamp Handling
    start_timestamp, end_timestamp = convert_to_unix(start_date), convert_to_unix(end_date)

    # Fetching Pool Data
    #pool_data = await pool_by_id(pool)

    # Fetching Hourly Price Data
    hourly_price_data = graphTwo(1, pool, start_timestamp, end_timestamp)
    #print(hourly_price_data)

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
        
        return pivot_fee_data(hourly_backtest, opt["priceToken"], investment_amount) if opt["period"] == "daily" else hourly_backtest

    
    
def chart1(dpd):
        # 1 Chart
    print(dpd)

    data=dpd[['date','feeToken0','feeToken1','fgV','feeV','feeUSD','amountV','activeLiquidity','amountTR','close']]
    data=data.fillna(0)
    data['date'] = pd.to_datetime(data['date'], format='%m/%d/%Y')

    temp =  data.resample('D',on='date').sum()
    final1=temp[['feeToken0','feeToken1','feeV','fgV','feeUSD']].copy()

    temp2 = data.resample('D',on='date').mean()
    final1['activeLiquidity']=temp2['activeLiquidity'].copy()

    temp3 = data.resample('D',on='date').first()
    final1[['amountV','amountTR']]=temp3[['amountV','amountTR']].copy()
    temp4 = data.resample('D',on='date').last()
    final1[['amountVlast']]=temp4[['amountV']]

    final1['S1%']=final1['feeV']/final1['amountV']*100#*365
    final1['unb%']=final1['fgV']/final1['amountTR']*100#*365
    final1['multiplier'] = final1['S1%'] / final1['unb%']
    final1['feeunb'] = final1['amountV']*final1['unb%']/100
    final1.to_csv("chart1.csv",sep = ";")
    
    print(final1[['feeunb','feeV','feeUSD','amountV','activeLiquidity','S1%','unb%']])
    apr = final1['feeV'].sum()/final1['amountV'].iloc[0]*365/len(final1.index)*100  
    apr_base = final1['feeunb'].sum()/final1['amountV'].iloc[0]*365/len(final1.index)*100

    print('------------------------------------------------------------------')
    print("Simulated Position returned", final1['feeV'].sum()/final1['amountV'].iloc[0]*100,"in ",len(final1.index)," days, for an APR of ",final1['feeV'].sum()/final1['amountV'].iloc[0]*365/len(final1.index)*100)
    print("Base position returned", final1['feeunb'].sum()/final1['amountV'].iloc[0]*100,"in ",len(final1.index)," days, for an APR of ",final1['feeunb'].sum()/final1['amountV'].iloc[0]*365/len(final1.index)*100)
    
    print ("Fees in token 0 and token 1",dpd['feeToken0'].sum(), dpd['feeToken1'].sum())
    print("Total Fees in USD", final1['feeUSD'].sum())
    print ('Your liquidity was active for:',final1['activeLiquidity'].mean())
    #print("Total fees earned based on initial deposit: ", total_fees_earned)
    print('------------------------------------------------------------------')
    return final1
        
        
        
        
    
    
if __name__ == "__main__":
    # Token 0: WBTC, Token 1: WETH
    Adress= "0xcbcdf9626bc03e24f779434178a73a0b4bad62ed" # WBTC ETH
    #Adress= "0x68f180fcce6836688e9084f035309e29bf0a2095" #  WBTC DAI
    #dpd= graphTwo(1,Adress,convert_to_unix("2023-05-25"),convert_to_unix("2023-12-24"))
    res = uniswap_strategy_backtest(Adress, 0.04177481929059751, 0.07653292116574624, "2023-05-25", "2023-12-24", investment_amount_usd=1000000, base=0)
    
    final1 = chart1(res)
    
