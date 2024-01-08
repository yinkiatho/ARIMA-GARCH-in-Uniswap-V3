import math
# from gql import gql, Client
# from gql.transport.requests import RequestsHTTPTransport
import pandas as pd
from datetime import datetime, timezone, timedelta
import numpy as np
from scipy.stats import norm
from numberstwo import *
import requests

def generate_bounds(start_volatility, end_volatility, start_price, expected_end=0.05269, confidence=0.95):
    # Calculate the lower and upper bounds
    z_stat = norm.ppf(1 - (1 - confidence) / 2)
    #print(z_stat)
    if start_price > expected_end:
        lower_bound = expected_end - end_volatility * z_stat
        upper_bound = start_price + start_volatility * z_stat
        
    else:
        lower_bound = start_price - start_volatility * z_stat
        upper_bound = expected_end + end_volatility * z_stat
    return lower_bound, upper_bound

def convert_to_unix(date_str):
    # Convert the date string to a datetime object
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    # Set the time to 8:00 AM
    date_obj = date_obj.replace(hour=8, minute=0, second=0, microsecond=0)
    unix_timestamp = int(date_obj.timestamp())

    return unix_timestamp
    
def convert_unix_to_datetime(unix_timestamp):
    try:
        # Convert Unix timestamp to datetime
        datetime_obj = datetime.utcfromtimestamp(unix_timestamp)
        return datetime_obj
    except Exception as e:
        print(f"Error converting Unix timestamp to datetime: {e}")
        return None


def graph(network,Adress,fromdate):

    if network == 1:

        sample_transport=RequestsHTTPTransport(
        url='https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3',
        verify=True,
        retries=5,
        )
        client = Client(
        transport=sample_transport
        )


    print(fromdate)

    query = gql('''
    query ($fromdate: Int!)
    {
    poolHourDatas(where:{pool:"'''+str(Adress)+'''",periodStartUnix_gt:$fromdate},orderBy:periodStartUnix,orderDirection:desc,first:1000)
    {
    periodStartUnix
    liquidity
    high
    low
    pool{
        
        totalValueLockedUSD
        totalValueLockedToken1
        totalValueLockedToken0
        token0
            {decimals
            }
        token1
            {decimals
            }
        }
    close
    feeGrowthGlobal0X128
    feeGrowthGlobal1X128
    }
 
    }
    ''')
    params = {
    "fromdate": fromdate
    }

    response = client.execute(query,variable_values=params)
    dpd =pd.json_normalize(response['poolHourDatas'])
    dpd=dpd.astype(float)
    return dpd



def graphTwo(network, Adress, fromdate, todate):
    if network == 1:
        sample_transport = RequestsHTTPTransport(
            url='https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3',
            verify=True,
            retries=5,
        )
        client = Client(
            transport=sample_transport
        )

    # Initialize an empty DataFrame to store all data
    all_data = pd.DataFrame()

    # Initial value for skip to start fetching data
    skip = 0

    while True:
        query = gql('''
        query ($fromdate: Int!, $todate: Int!, $skip: Int)
        {
            poolHourDatas(where: {pool: "'''+str(Adress)+'''", periodStartUnix_gt: $fromdate, periodStartUnix_lt: $todate}, orderBy: periodStartUnix, orderDirection: desc, skip: $skip, first: 1000)
            {
            periodStartUnix
            tick
            liquidity
            high
            low
            pool{
        
                totalValueLockedUSD
                totalValueLockedToken1
                totalValueLockedToken0
                token0
                    {decimals
                    }
                token1
                    {decimals
                    }
                }
            close
            feeGrowthGlobal0X128
            feeGrowthGlobal1X128
            }
            }
        ''')

        params = {
            "fromdate": fromdate,
            "todate": todate,
            "skip": skip
        }

        response = client.execute(query, variable_values=params)

        # Extract data from the response
        pool_hour_datas = response.get('poolHourDatas', [])
        if not pool_hour_datas:
            break  # No more data to fetch

        # Extract and append data to the DataFrame
        chunk_data = pd.json_normalize(pool_hour_datas, sep='.')
        chunk_data = chunk_data.astype(float)
        all_data = pd.concat([all_data, chunk_data], ignore_index=True)

        # Update the skip for the next iteration
        skip += 1000  # Assuming you want to skip 1000 records at a time

    return all_data


def chart1(dpd,base,myliquidity, initial_amounts):
    
    results = {}
    
    amount_deposit_0, amount_deposit_1 = initial_amounts
    print(f'Amount Depositied WBTC: {amount_deposit_0}, Amount Deposited WETH: {amount_deposit_1}')
    initial_amountV = amount_deposit_0 + (amount_deposit_1 * dpd['close'].iloc[-1])
    print(f'Initial Amount in WBTC: {initial_amountV}')
    decimal0=dpd.iloc[0]['pool.token0.decimals']
    decimal1=dpd.iloc[0]['pool.token1.decimals']
    

    if base==0:
        dpd['feeV']= (dpd['myfee0'] ) + (dpd['myfee1'] * dpd['close'])
        dpd['amountV']= (dpd['amount0'] ) + (dpd['amount1'] * dpd['close'])
        dpd['amountunb']= (dpd['amount0unb'] ) + (dpd['amount1unb']* dpd['close'])
        dpd['fgV']= (dpd['fee0token']) + (dpd['fee1token']* dpd['close'])
        dpd['feeusd']= dpd['feeV'] * (dpd['pool.totalValueLockedUSD'].iloc[0] / (dpd['pool.totalValueLockedToken1'].iloc[0]* dpd['close'].iloc[0]+(dpd['pool.totalValueLockedToken0'].iloc[0])))


    else:

        dpd['feeV']= (dpd['myfee0'] / dpd['close']) + dpd['myfee1']
        dpd['amountV']= (dpd['amount0'] / dpd['close'])+ dpd['amount1']
        dpd['feeVbase0']= dpd['myfee0'] + (dpd['myfee1']* dpd['close'])
        dpd['amountunb']= (dpd['amount0unb'] / dpd['close'])+ dpd['amount1unb']
        dpd['fgV']=(dpd['fee0token'] / dpd['close'])+ dpd['fee1token']
        dpd['feeusd']= dpd['feeV'] * ( dpd['pool.totalValueLockedUSD'].iloc[0] / (dpd['pool.totalValueLockedToken1'].iloc[0] + (dpd['pool.totalValueLockedToken0'].iloc[0]/dpd['close'].iloc[0])))
        

    dpd['date']=pd.to_datetime(dpd['periodStartUnix'],unit='s')

    # 1 Chart
    
    #dpd['fgV']= (dpd['fg0'] / dpd['close'].iloc[0] + dpd['fg1'])
    #rint(dpd['fg1']/dpd['amount1unb'])

    data=dpd[['date','myfee0','myfee1','fgV','feeV','feeusd','amountV','ActiveLiq','amountunb','amount0','amount1','close']]
    data=data.fillna(0)

    temp = data.resample('D',on='date').sum()
    final1 = temp[['myfee0','myfee1','feeV','fgV','feeusd']].copy()

    temp2 = data.resample('D',on='date').mean()
    final1['ActiveLiq']=temp2['ActiveLiq'].copy()
    
    temp3 = data.resample('D',on='date').first()
    final1[['amountV','amountunb']]=temp3[['amountV','amountunb']].copy()
    temp4 = data.resample('D',on='date').last()
    final1[['amountVlast']]=temp4[['amountV']]

    final1['S1%']=final1['feeV']/final1['amountV']*100#*365
    final1['unb%']=final1['fgV']/final1['amountunb']*100#*365
    final1['multiplier'] = final1['S1%'] / final1['unb%']
    final1['feeunb'] = final1['amountV']*final1['unb%']/100
    final1.to_csv("chart1.csv",sep = ";")
    
    print(final1[['feeunb','feeV','feeusd','amountV','ActiveLiq','S1%','unb%']])

    print('------------------------------------------------------------------')
    print("Simulated Position returned", final1['feeV'].sum()/final1['amountV'].iloc[0]*100,"in ",len(final1.index)," days, for an APR of ",final1['feeV'].sum()/final1['amountV'].iloc[0]*365/len(final1.index)*100)
    print("Base position returned", final1['feeunb'].sum()/final1['amountV'].iloc[0]*100,"in ",len(final1.index)," days, for an APR of ",final1['feeunb'].sum()/final1['amountV'].iloc[0]*365/len(final1.index)*100)
    
    print ("Fees in token 0 and token 1",dpd['myfee0'].sum(), dpd['myfee1'].sum())
    print("Total Fees in USD", final1['feeusd'].sum())
    print ('Your liquidity was active for:',final1['ActiveLiq'].mean())

    #print("Total fees earned based on initial deposit: ", total_fees_earned)
    print('------------------------------------------------------------------')
    
    results['Position Return'] = final1['feeV'].sum()/final1['amountV'].iloc[0]*100
    results['Base Return'] = final1['feeunb'].sum()/final1['amountV'].iloc[0]*100
    results['Position APR'] = final1['feeV'].sum()/final1['amountV'].iloc[0]*365/len(final1.index)*100
    results['Base APR'] = final1['feeunb'].sum()/final1['amountV'].iloc[0]*365/len(final1.index)*100
    results['Fees in Token 0'] = dpd['myfee0'].sum()
    results['Fees in Token 1'] = dpd['myfee1'].sum()
    results['Total Fees in USD'] = final1['feeusd'].sum()
    results['Mean Percentage of Active Liquidity'] = final1['ActiveLiq'].mean()
    
    
    final2=temp3[['amountV','amount0','amount1','close']].copy()
    final2['feeV']=temp['feeV'].copy()
    final2[['amountVlast']]=temp4[['amountV']]
    

    final2['HODL']=final2['amount0'].iloc[0] / final2['close'] + final2['amount1'].iloc[0]
    
    final2['IL']=final2['amountVlast']- final2['HODL']
    final2['ActiveLiq']=temp2['ActiveLiq'].copy()
    final2['feecumsum']=final2['feeV'].cumsum()
    final2 ['PNL']= final2['feecumsum'] + final2['IL']#-Bfinal['gas']

    final2['HODLnorm']=final2['HODL']/final2['amountV'].iloc[0]*100
    final2['ILnorm']=final2['IL']/final2['amountV'].iloc[0]*100
    final2['PNLnorm']=final2['PNL']/final2['amountV'].iloc[0]*100
    final2['feecumsumnorm'] = final2['feecumsum']/final2['amountV'].iloc[0]*100
    ch2=final2[['amountV','feecumsum']]
    ch3=final2[['ILnorm','PNLnorm','feecumsumnorm']]

    final2.to_csv("chart2.csv",sep = ";")
    #print(ch2)
    #print(ch3)

    #final3=data
    final3=pd.DataFrame()
    final3['amountV']=data['amountV']

    final3['amountVlast']=data['amountV'].shift(-1)
    final3['date']=data['date']
    final3['HODL']=data['amount0'].iloc[0] / data['close'] + data['amount1'].iloc[0]

    final3['amountVlast'].iloc[-1]=final3['HODL'].iloc[-1]
    final3['IL']=final3['amountVlast']- final3['HODL']
    final3['feecumsum']=data['feeV'][::-1].cumsum()
    final3 ['PNL']= final3['feecumsum'] + final3['IL']
    final3['HODLnorm']=final3['HODL']/final3['amountV'].iloc[0]*100
    final3['ILnorm']=final3['IL']/final3['amountV'].iloc[0]*100
    final3['PNLnorm']=final3['PNL']/final3['amountV'].iloc[0]*100
    final3['feecumsumnorm'] = final3['feecumsum']/final3['amountV'].iloc[0]*100

    ch2=final3[['amountV','feecumsum']]
    ch3=final3[['ILnorm','PNLnorm','feecumsumnorm']]


    #print(ch2)
    #print(ch3)
    
    return final1,final2,final3, results



def get_amount0(sqrtA,sqrtB,liquidity,decimals):
    
    if (sqrtA > sqrtB):
          (sqrtA,sqrtB)=(sqrtB,sqrtA)
    
    amount0=((liquidity*2**96*(sqrtB-sqrtA)/sqrtB/sqrtA)/10**decimals)
    
    return amount0

def get_amount1(sqrtA,sqrtB,liquidity,decimals):
    
    if (sqrtA > sqrtB):
        (sqrtA,sqrtB)=(sqrtB,sqrtA)
    
    amount1=liquidity*(sqrtB-sqrtA)/2**96/10**decimals
    
    return amount1

def get_amounts(asqrt,asqrtA,asqrtB,liquidity,decimal0,decimal1):

    sqrt=(np.sqrt(asqrt*10**(decimal1-decimal0)))*(2**96)
    sqrtA=np.sqrt(asqrtA*10**(decimal1-decimal0))*(2**96)
    sqrtB=np.sqrt(asqrtB*10**(decimal1-decimal0))*(2**96)

    if (sqrtA > sqrtB):
        (sqrtA,sqrtB)=(sqrtB,sqrtA)

    if sqrt<=sqrtA:

        amount0=get_amount0(sqrtA,sqrtB,liquidity,decimal0)
        return amount0,0
   
    elif sqrt<sqrtB and sqrt>sqrtA:
        amount0=get_amount0(sqrt,sqrtB,liquidity,decimal0)
   
        amount1=get_amount1(sqrtA,sqrt,liquidity,decimal1)
       
        return amount0,amount1
    
    else:
        amount1=get_amount1(sqrtA,sqrtB,liquidity,decimal1)
        return 0,amount1      



'''get_liquidity function'''
#Use 'get_liquidity' function to calculate liquidity as a function of amounts and price range
def get_liquidity0(sqrtA,sqrtB,amount0,decimals):
    if (sqrtA > sqrtB):
          (sqrtA,sqrtB)=(sqrtB,sqrtA)
    
    liquidity=amount0/((2**96*(sqrtB-sqrtA)/sqrtB/sqrtA)/10**decimals)
    return liquidity

def get_liquidity1(sqrtA,sqrtB,amount1,decimals):
    
    if (sqrtA > sqrtB):
        (sqrtA,sqrtB)=(sqrtB,sqrtA)
    
    liquidity=amount1/((sqrtB-sqrtA)/2**96/10**decimals)
    return liquidity

def get_liquidity(asqrt,asqrtA,asqrtB,amount0,amount1,decimal0,decimal1):
    
        sqrt=(np.sqrt(asqrt*10**(decimal1-decimal0)))*(2**96)
        sqrtA=np.sqrt(asqrtA*10**(decimal1-decimal0))*(2**96)
        sqrtB=np.sqrt(asqrtB*10**(decimal1-decimal0))*(2**96)

        
        if (sqrtA > sqrtB):
            (sqrtA,sqrtB)=(sqrtB,sqrtA)
    
        if sqrt<=sqrtA:
            
            liquidity0=get_liquidity0(sqrtA,sqrtB,amount0,decimal0)
            return liquidity0
        elif sqrt<sqrtB and sqrt>sqrtA:
           
            liquidity0=get_liquidity0(sqrt,sqrtB,amount0,decimal0)
            liquidity1=get_liquidity1(sqrtA,sqrt,amount1,decimal1)
            liquidity=liquidity0 if liquidity0<liquidity1 else liquidity1
            return liquidity
        
        else:
            liquidity1=get_liquidity1(sqrtA,sqrtB,amount1,decimal1)
            return liquidity1
        
        
def calc_unbounded_fees(globalfee0, prev_globalfee0, globalfee1, prev_globalfee1, decimal0, decimal1):
    fg0_0 = (int(globalfee0) / math.pow(2, 128)) / (math.pow(10, decimal0))
    fg0_1 = (int(prev_globalfee0) / math.pow(2, 128)) / (math.pow(10, decimal0))
    fg1_0 = (int(globalfee1) / math.pow(2, 128)) / (math.pow(10, decimal1))
    fg1_1 = (int(prev_globalfee1) / math.pow(2, 128)) / (math.pow(10, decimal1))
    
    fg0 = fg0_0 - fg0_1
    fg1 = fg1_0 - fg1_1
    
    return [fg0, fg1]

def get_tick_from_price(price, decimal0, decimal1, base_selected=0):
    
    val_to_log = float(price) * math.pow(10, (decimal0 - decimal1))
    tick_idx_raw = math.log(val_to_log) / math.log(1.0001)
    
    return round(tick_idx_raw, 0)

def active_liquidity_for_candle(min_val, max_val, low, high):
    divider = (high - low) if (high - low) != 0 else 1
    ratio_true = (min(max_val, high) - max(min_val, low)) / divider if (high > min_val) and (low < max_val) else 1
    ratio = ratio_true * 100 if (high > min_val) and (low < max_val) else 0

    return round(ratio, 2) if not math.isnan(ratio) and ratio else 0

def tokens_from_liquidity(price, low, high, liquidity, decimal0, decimal1):
    decimal = decimal1 - decimal0
    low_high = [math.sqrt(low * math.pow(10, decimal)) * math.pow(2, 96), math.sqrt(high * math.pow(10, decimal)) * math.pow(2, 96)]
    s_price = (math.sqrt(price * math.pow(10, decimal))) * math.pow(2, 96)
    s_low = min(low_high)
    s_high = max(low_high)
    
    if s_price <= s_low:
        amount1 = (liquidity * math.pow(2, 96) * (s_high - s_low) / s_high / s_low) / math.pow(10, decimal0)
        return [0, amount1]
    elif s_low < s_price < s_high:
        amount0 = liquidity * (s_price - s_low) / math.pow(2, 96) / math.pow(10, decimal1)
        amount1 = (liquidity * math.pow(2, 96) * (s_high - s_price) / s_high / s_price) / math.pow(10, decimal0)
        return [amount0, amount1]
    else:
        amount0 = liquidity * (s_high - s_low) / math.pow(2, 96) / math.pow(10, decimal1)
        return [amount0, 0]

def tokens_for_strategy(min_range, max_range, investment, price, decimal):
    sqrt_price = math.sqrt(price * math.pow(10, decimal))
    sqrt_low = math.sqrt(min_range * math.pow(10, decimal))
    sqrt_high = math.sqrt(max_range * math.pow(10, decimal))

    if sqrt_low < sqrt_price < sqrt_high:
        delta = investment / ((sqrt_price - sqrt_low) + ((1 / sqrt_price - 1 / sqrt_high) * (price * math.pow(10, decimal))))
        amount1 = delta * (sqrt_price - sqrt_low)
        amount0 = delta * ((1 / sqrt_price) - (1 / sqrt_high)) * math.pow(10, decimal)
    elif sqrt_price <= sqrt_low:
        delta = investment / (((1 / sqrt_low) - (1 / sqrt_high)) * price)
        amount1 = 0
        amount0 = delta * ((1 / sqrt_low) - (1 / sqrt_high))
    else:
        delta = investment / (sqrt_high - sqrt_low)
        amount1 = delta * (sqrt_high - sqrt_low)
        amount0 = 0

    return [amount0, amount1]


def liquidity_for_strategy(price, low, high, tokens0, tokens1, decimal0, decimal1):
    decimal = decimal1 - decimal0
    low_high = [
        (math.sqrt(low * 10**decimal) * 2**96),
        (math.sqrt(high * 10**decimal) * 2**96)
    ]

    s_price = (math.sqrt(price * 10**decimal) * 2**96)
    s_low = min(low_high)
    s_high = max(low_high)

    if s_price <= s_low:
        return tokens0 / (
            (2**96 * (s_high - s_low) / s_high / s_low) / 10**decimal0
        )
    elif s_low < s_price <= s_high:
        liq0 = tokens0 / (
            (2**96 * (s_high - s_price) / s_high / s_price) / 10**decimal0
        )
        liq1 = tokens1 / ((s_price - s_low) / 2**96 / 10**decimal1)
        return min(liq1, liq0)
    else:
        return tokens1 / (
            (s_high - s_low) / 2**96 / 10**decimal1)
            
            
            
            
def calc_fees(data, decimal0, decimal1, price_token, liquidity, unbounded_liquidity, investment, _min, _max):
    result = []
    #print(data)
    latest_rec = data.iloc[-1]

    for i, d in data.iterrows():
        #print(d)
        fg = [0, 0] if i - 1 < 0 else calc_unbounded_fees(
            d['feeGrowthGlobal0X128'],
            data.iloc[i - 1]['feeGrowthGlobal0X128'],
            d['feeGrowthGlobal1X128'],
            data.iloc[i - 1]['feeGrowthGlobal1X128'],
            decimal0, decimal1
        )

        low = d['low'] 
        high = d['high'] 

        low_tick = get_tick_from_price(low, decimal0, decimal1, price_token)
        high_tick = get_tick_from_price(high, decimal0, decimal1, price_token)
        min_tick = get_tick_from_price(_min, decimal0, decimal1, price_token)
        max_tick = get_tick_from_price(_max, decimal0, decimal1, price_token)

        active_liquidity = active_liquidity_for_candle(min_tick, max_tick, low_tick, high_tick)
        tokens = tokens_from_liquidity(
            1 / float(d['close']) if price_token == 1 else float(d['close']),
            _min, _max, liquidity,
            decimal0, decimal1
        )
        fee_token_0 = 0 if i == 0 else fg[0] * liquidity * active_liquidity / 100
        fee_token_1 = 0 if i == 0 else fg[1] * liquidity * active_liquidity / 100

        fee_unb_0 = 0 if i == 0 else fg[0] * unbounded_liquidity
        fee_unb_1 = 0 if i == 0 else fg[1] * unbounded_liquidity

        fg_v, fee_v, fee_unb, amount_v, fee_usd, amount_tr = 0, 0, 0, 0, 0, 0
        
        first_close = 1 / float(data.iloc[0]['close']) if price_token == 1 else float(data.iloc[0]['close'])

        token_ratio_first_close = tokens_from_liquidity(
            first_close, _min, _max, liquidity,
            decimal0, decimal1
        )
        x0, y0 = token_ratio_first_close[1], token_ratio_first_close[0]

        if price_token == 0:
            fg_v = 0 if i == 0 else fg[0] + fg[1] * float(d['close'])
            fee_v = 0 if i == 0 else fee_token_0 + fee_token_1 * float(d['close'])
            fee_unb = 0 if i == 0 else fee_unb_0 + fee_unb_1 * float(d['close'])
            amount_v = tokens[0] + tokens[1] * float(d['close'])
            fee_usd = fee_v * float(latest_rec['pool.totalValueLockedUSD']) / (
                    float(latest_rec['pool.totalValueLockedToken1']) * float(latest_rec['close']) +
                    float(latest_rec['pool.totalValueLockedToken0'])
            )
            amount_tr = investment + (amount_v - (x0 * float(d['close']) + y0))
        elif price_token == 1:
            fg_v = 0 if i == 0 else fg[0] / float(d['close']) + fg[1]
            fee_v = 0 if i == 0 else fee_token_0 / float(d['close']) + fee_token_1
            fee_unb = 0 if i == 0 else fee_unb_0 + fee_unb_1 * float(d['close'])
            amount_v = tokens[1] / float(d['close']) + tokens[0]
            fee_usd = fee_v * float(latest_rec['pool.totalValueLockedUSD']) / (
                    float(latest_rec['pool.totalValueLockedToken1']) +
                    float(latest_rec['pool.totalValueLockedToken0']) / float(latest_rec['close'])
            )
            amount_tr = investment + (amount_v - (x0 * (1 / float(d['close'])) + y0))

        date = datetime.utcfromtimestamp(d['periodStartUnix'])
        result.append({
            **d,
            'day': date.day,
            'month': date.month,
            'year': date.year,
            'fg0': fg[0],
            'fg1': fg[1],
            'activeLiquidity': active_liquidity,
            'feeToken0': fee_token_0,
            'feeToken1': fee_token_1,
            'tokens': tokens,
            'fgV': fg_v,
            'feeV': fee_v,
            'feeUnb': fee_unb,
            'amountV': amount_v,
            'amountTR': amount_tr,
            'feeUSD': fee_usd,
            'close': float(d['close']),
            'baseClose': 1 / float(d['close']) if price_token == 1 else float(d['close'])
        })

    return result




def create_pivot_record(date, data, price_token=0):
    #print(data)
    return {
        'date': f'{date.month}/{date.day}/{date.year}',
        'day': date.day,
        'month': date.month,
        'year': date.year,
        'feeToken0': data['feeToken0'],
        'feeToken1': data['feeToken1'],
        'feeV': data['feeV'],
        'feeUnb': data['feeUnb'],
        'fgV': float(data['fgV']),
        'feeUSD': data['feeUSD'],
        'activeLiquidity': 0 if math.isnan(data['activeLiquidity']) else data['activeLiquidity'],
        'amountV': data['amountV'],
        'amountTR': data['amountTR'],
        'amountVLast': data['amountV'],
        'percFee': data['feeV'] / data['amountV'] * 100,
        'close': float(data['close']),
        'baseClose': 1 / float(data['close']) if price_token == 1 else float(data['close']),
        'count': 1
    }

def pivot_fee_data(data, price_token, investment):
    first_date = datetime.utcfromtimestamp(data[0]['periodStartUnix']).replace(tzinfo=timezone.utc)
    pivot = [create_pivot_record(first_date, data[0])]


    for i, d in enumerate(data[1:], start=1):
        current_date = datetime.utcfromtimestamp(d['periodStartUnix']).replace(tzinfo=timezone.utc)
        current_price_tick = pivot[-1]
        #print(current_price_tick)
        #print('00')
        #print(d)

        if (current_date.day, current_date.month, current_date.year) == (
            current_price_tick['day'],
            current_price_tick['month'],
            current_price_tick['year']
        ):
            current_price_tick['feeToken0'] += d['feeToken0']
            current_price_tick['feeToken1'] += d['feeToken1']
            current_price_tick['feeV'] += d['feeV']
            current_price_tick['feeUnb'] += d['feeUnb']
            current_price_tick['fgV'] += float(d['fgV'])
            current_price_tick['feeUSD'] += d['feeUSD']
            current_price_tick['activeLiquidity'] += d['activeLiquidity']
            current_price_tick['amountVLast'] = d['amountV']
            current_price_tick['count'] += 1

            if i == len(data) - 1:
                current_price_tick['activeLiquidity'] /= current_price_tick['count']
                current_price_tick['percFee'] = current_price_tick['feeV'] / current_price_tick['amountV'] * 100
        else:
            current_price_tick['activeLiquidity'] /= current_price_tick['count']
            current_price_tick['percFee'] = current_price_tick['feeV'] / current_price_tick['amountV'] * 100
            pivot.append(create_pivot_record(current_date, d))
            
    pivot_df = pd.DataFrame(pivot)
    
    return pivot_df


def date_by_days_ago(days, end_date=None):
    date = datetime.utcfromtimestamp(end_date) if end_date is not None else datetime.utcnow()
    result_date = date - timedelta(days=days)
    return int(result_date.timestamp())



def chart1(dpd):
        # 1 Chart
    #print(dpd)

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
    
    #print(final1[['feeunb','feeV','feeUSD','amountV','activeLiquidity','S1%','unb%']])
    apr = final1['feeV'].sum()/final1['amountV'].iloc[0]*365/len(final1.index)*100  
    apr_base = final1['feeunb'].sum()/final1['amountV'].iloc[0]*365/len(final1.index)*100
    fees_usd = final1['feeUSD'].sum()
    final_net_liquidity = final1['amountV'].iloc[-1]
    active_liquidity = final1['activeLiquidity'].mean()

    #print('------------------------------------------------------------------')
    print("Simulated Position returned", final1['feeV'].sum()/final1['amountV'].iloc[0]*100,"in ",len(final1.index)," days, for an APR of ",final1['feeV'].sum()/final1['amountV'].iloc[0]*365/len(final1.index)*100)
    print("Base position returned", final1['feeunb'].sum()/final1['amountV'].iloc[0]*100,"in ",len(final1.index)," days, for an APR of ",final1['feeunb'].sum()/final1['amountV'].iloc[0]*365/len(final1.index)*100)
    
    print ("Fees in token 0 and token 1",dpd['feeToken0'].sum(), dpd['feeToken1'].sum())
    print("Total Fees in USD", final1['feeUSD'].sum())
    print ('Your liquidity was active for:',final1['activeLiquidity'].mean())
    print('Final Net Liquidity Value of LP Investment (WBTC): ', final1['amountV'].iloc[-1])
    #print("Total fees earned based on initial deposit: ", total_fees_earned)
    #print('------------------------------------------------------------------')
    return final1, [fees_usd, apr, apr_base, final_net_liquidity, active_liquidity]


'''Input
- Current Close price of pool
- Risk Free Rate (staking APY)
- Strike Price (Call/Put)
- Type of Option (Call/Put)
- Time to Expiration (in years)
'''


def black_scholes_call(S0, X, T, r, sigma): 
    """ 
    Calculate the Black-Scholes option price for a European call option. 

    :param S0: Current Close Price of Uniswap v3 Pool 
    :param X: Strike price of the option 
    :param T: Time to expiration in years 
    :param r: Risk-free interest rate 
    :param sigma: Volatility of the stock's returns 
    :return: Call option price 
    """ 
    d1 = (math.log(S0 / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T)) 
    d2 = d1 - sigma * math.sqrt(T) 
    call_price = (S0 * norm.cdf(d1) - X * math.exp(-r * T) * norm.cdf(d2)) 
    return call_price

def black_scholes_put(S0, X, T, r, sigma): 
    """ 
    Calculate the Black-Scholes option price for a European put option. 

    :param S0: Current Close Price of Uniswap v3 Pool 
    :param X: Strike price of the option 
    :param T: Time to expiration in years 
    :param r: Risk-free interest rate 
    :param sigma: Volatility of the stock's returns 
    :return: Call option price 
    """ 
    d1 = (np.log(S0 / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)) 
    d2 = d1 - sigma * np.sqrt(T) 
    put_price = X*math.exp(-r*T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return put_price

def implied_volatility(C,S,K,r,T):
    """ 
    Calculate the Implied Volatility from option price for a European call option. 

    :param C: Call Option Price 
    :param S: Current Close Price of Uniswap v3 Pool 
    :param K: Strike price of the option 
    :param T: Time to expiration in years 
    :param r: Risk-free interest rate 
    :return: implied Volatility 
    """ 
    tolerance = 0.001
    epsilon = 1
    
    count = 0
    max_iter = 1000

    vol = 0.50 
    while epsilon>tolerance:
        count += 1
        if count>=max_iter:
            print("Max iterations reached!")
            break
        original_vol = vol

        call_price = black_scholes_call(S,K,T,r,vol)
        function_value = call_price - C
        d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T)) 
        vega = S*norm.pdf(d1) * math.sqrt(T)
        vol = -function_value/vega + vol

        epsilon = abs((vol - original_vol)/original_vol)
    print(vol)
    return vol


def optionPrice(coin, strike, time, type, coinPrice, rf):
    url = f'https://www.deribit.com/api/v2/public/get_instruments?currency={coin}&kind=option'
    instrument_name = requests.get(url)
    instrument_name = instrument_name.json()['result'][0]["instrument_name"]    #eg of instrument_name: "BTC-9JAN24-38500-C"
    details_of_option = instrument_name.split(sep='-')
    strike_temp = details_of_option[2]
    time_temp = 1/365 #Time = 1 day expiration
    #Picked the first option in the order book for the coin specified and the name is noted with strike, Expiration date


    response = requests.get(f'https://www.deribit.com/api/v2/public/get_order_book?depth=5&instrument_name={instrument_name}')
    response = response.json()
    # print(response)
    coinPrice_temp = response["result"]["underlying_price"]
    C = response["result"]["mark_price"]
    #Getting underlying price and option price

    # implied_vol = implied_volatility(C,coinPrice_temp,strike_temp,rf,time_temp)
    # Calculaing Implied volatility
    implied_vol = response["result"]["mark_iv"]/100
    print(implied_vol)

    #Using Implied Volatility to calculate option price
    if(type=='CALL'):
        print((coinPrice,strike,time,rf,implied_vol))
        callOptionPrice = black_scholes_call(coinPrice,strike,time,rf,implied_vol)
        print(callOptionPrice)
        return callOptionPrice
    else:
        putOptionPrice = black_scholes_put(coinPrice,strike,time,rf,implied_vol)
        print(putOptionPrice)
        return putOptionPrice
    
def generate_hodl(prices, initial_investment):
    btc_start, btc_end, eth_start, eth_end = prices
    amount0, amount1 = initial_investment
        
    # Calculate the final value for each coin based on the price change
    btc_final_value = amount0 * (btc_end)
    eth_final_value = amount1 * (eth_end)
        
    # Calculate the total final value in USD
    total_final_value = btc_final_value + eth_final_value
        
    return total_final_value


# import math
# from scipy.stats import norm

# S = 1#Price of Underlying
# K = 1#Strike Price
# T = 1# Time to maturity
# r = 1# Use 10 year Treasury Yield
# Vol = 1# Volatility

# d1 = (math.log(S/K) + (r+0.5*Vol**2)*T)/(Vol*math.sqrt(T))
# d2 = d1 - Vol*math.sqrt(T)

# C = S * norm.cdf(d1) - K*math.exp(-r*T) * norm.cdf(d2)
# P = K*math.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
optionPrice("BTC", 100, 1, 'CALL', 42000, 2)