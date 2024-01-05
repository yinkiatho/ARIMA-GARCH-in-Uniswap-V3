from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

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


def chart1(dpd,base,myliquidity):

    if base==0:
        dpd['feeV']= (dpd['myfee0'] ) + (dpd['myfee1']* dpd['close'])
        dpd['amountV']= (dpd['amount0'] ) + (dpd['amount1']* dpd['close'])
        dpd['amountunb']= (dpd['amount0unb'] )+ (dpd['amount1unb']* dpd['close'])
        dpd['fgV']= (dpd['fee0token'])+ (dpd['fee1token']* dpd['close'])
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

    temp =  data.resample('D',on='date').sum()
    final1=temp[['myfee0','myfee1','feeV','fgV','feeusd']].copy()

    temp2 = data.resample('D',on='date').mean()
    final1['ActiveLiq']=temp2['ActiveLiq'].copy()
    
    temp3 = data.resample('D',on='date').first()
    final1[['amountV','amountunb']]=temp3[['amountV','amountunb']].copy()
    temp4 = data.resample('D',on='date').last()
    final1[['amountVlast']]=temp4[['amountV']]

    final1['S1%']=final1['feeV']/final1['amountV']*100#*365
    final1['unb%']=final1['fgV']/final1['amountunb']*100#*365
    final1['multiplier']=final1['S1%']/final1['unb%']
    final1['feeunb'] = final1['amountV']*final1['unb%']/100
    final1.to_csv("chart1.csv",sep = ";")
    
    print(final1[['feeunb','feeV','feeusd','amountV','ActiveLiq','S1%','unb%','ActiveLiq']])

    print('------------------------------------------------------------------')
    print("Simulated Position returned", final1['feeV'].sum()/final1['amountV'].iloc[0]*100,"in ",len(final1.index)," days, for an APR of ",final1['feeV'].sum()/final1['amountV'].iloc[0]*365/len(final1.index)*100)
    print("Base position returned", final1['feeunb'].sum()/final1['amountV'].iloc[0]*100,"in ",len(final1.index)," days, for an APR of ",final1['feeunb'].sum()/final1['amountV'].iloc[0]*365/len(final1.index)*100)
    
    print ("Fees in token 1 and token 2",dpd['myfee0'].sum(),dpd['myfee1'].sum() )
    print("TotalFees in USD", final1['feeusd'].sum())
    print ('Your liquidity was active for:',final1['ActiveLiq'].mean())
    
    if base == 0:
        forecast= (dpd['feeV'].sum()*myliquidity*final1['ActiveLiq'].mean())	
        print(dpd['feeV'])
        
    else:
        forecast= (dpd['feeVbase0'].sum()*myliquidity*final1['ActiveLiq'].mean())
        print(dpd['feeVbase0'])
        
    
    print('forecast: ',forecast)
    print('------------------------------------------------------------------')
    # 1 chart e' completo
    
    # 2 chart

    
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
    print(ch2)
    print(ch3)

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


    print(ch2)
    print(ch3)
    
    return final1,final2,final3



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