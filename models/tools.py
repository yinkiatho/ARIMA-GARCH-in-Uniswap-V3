import math
import requests
import pandas as pd
import numpy as np
from scipy.stats import norm


def ILoss_calculator(iv0, iv1, startingp0, startingp1, curr_p1, curr_p0):
    return abs(1 + (
        iv0*((curr_p0 - startingp0)/(2*startingp0) - (curr_p1 - startingp1)/(2*startingp1)) + iv1*((curr_p1 - startingp1)/(2*startingp1) - (curr_p0 - startingp0)/(2*startingp0))
        )/(iv0 + iv1))
    
    
    
def position_value():
    return 


def x_virtual_reserve(Liquidity, p_lower, p_upper):
    return math.sqrt(Liquidity*math.sqrt(p_lower) / math.sqrt(p_upper))


def y_virtual_reserve(Liquidity, p_lower, p_upper):
    return math.sqrt(Liquidity/math.sqrt(p_lower) * math.sqrt(p_upper))


def fetch_tick_day_datas(start_date_unix_timestamp=1684800000, tick_idx_gte=100, tick_idx_lte=200):
    graphql_endpoint = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"
    
    
    pool_id = "0xcbcdf9626bc03e24f779434178a73a0b4bad62ed"
    
    graphql_query = f"""
    {{
      tickDayDatas(
        where: {{
          date_in: [{start_date_unix_timestamp}],
          pool_in: ["{pool_id}"],
        
        }},
        skip: 0,
        first: 1000,
        orderBy: date,
        orderDirection: asc
      ) {{
        id
        date
        pool {{
          id
          token0 {{
            id
            symbol
          }}
          token1 {{
            id
            symbol
          }}
        }}
        tick {{
          id
          tickIdx
        }}
        liquidityGross
        liquidityNet
        volumeToken0
        volumeToken1
        volumeUSD
        feesUSD
        feeGrowthOutside0X128
        feeGrowthOutside1X128
      }}
}}
    """
  
    headers = {
        'Content-Type': 'application/json',
    }

    data = {
        'query': graphql_query,
    }

    response = requests.post(graphql_endpoint, headers=headers, json=data)

    if response.status_code == 200 and response.json().get('data'):
        output = response.json()
        #print(output)
        df = pd.json_normalize(output['data']['tickDayDatas'])
        return df
    else:
        print(f"Error: {response.status_code}")
        print(response.text)  # Print the response content for debugging
        return None
      
      
'''liquitidymath'''
'''Python library to emulate the calculations done in liquiditymath.sol of UNI_V3 peryphery contract'''

#sqrtP: format X96 = int(1.0001**(tick/2)*(2**96))
#liquidity: int
#sqrtA = price for lower tick
#sqrtB = price for upper tick
'''get_amounts function'''
#Use 'get_amounts' function to calculate amounts as a function of liquitidy and price range
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

def get_amounts(tick,tickA,tickB,liquidity,decimal0,decimal1):

    sqrt=(1.0001**(tick/2)*(2**96))
    sqrtA=(1.0001**(tickA/2)*(2**96))
    sqrtB=(1.0001**(tickB/2)*(2**96))

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

'''get token amounts relation'''
#Use this formula to calculate amount of t0 based on amount of t1 (required before calculate liquidity)
#relation = t1/t0      
def amounts_relation(tick,tickA,tickB,decimals0,decimals1):
    
    sqrt=(1.0001**tick/10**(decimals1-decimals0))**(1/2)
    sqrtA=(1.0001**tickA/10**(decimals1-decimals0))**(1/2)
    sqrtB=(1.0001**tickB/10**(decimals1-decimals0))**(1/2)
    
    if sqrt==sqrtA or sqrt==sqrtB:
        relation=0
       

    relation=(sqrt-sqrtA)/((1/sqrt)-(1/sqrtB))     
    return relation       



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

def get_liquidity(tick,tickA,tickB,amount0,amount1,decimal0,decimal1):
    
        sqrt=(1.0001**(tick/2)*(2**96))
        sqrtA=(1.0001**(tickA/2)*(2**96))
        sqrtB=(1.0001**(tickB/2)*(2**96))
        
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
          
          
          
def black_scholes(S0, X, T, r, sigma): 
    """ 
    Calculate the Black-Scholes option price for a European call option. 

    :param S0: Current Close Price of Uniswap v3 Pool 
    :param X: Strike price of the option 
    :param T: Time to expiration in years 
    :param r: Risk-free interest rate 
    :param sigma: Volatility of the stock's returns 
    :return: Call option price 
    """ 
    d1 = (np.log(S0 / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)) 
    d2 = d1 - sigma * np.sqrt(T) 
    call_price = (S0 * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)) 
    return call_price 
 
def calculate_strike_prices(S0, r, T, sigma, min_range=-50, max_range=120, step=10): 
    """ 
    Calculate call option prices for a range of strike prices. 
 
    :param S0: Current Close Price of Uniswap v3 Pool 
    :param r: Staking APY
    :param T: Time to expiration in years (D / 365)
    :param sigma: Forecasted daily volatility of the Pool returns
    :param min_range: Minimum percentage range for strike price 
    :param max_range: Maximum percentage range for strike price 
    :param step: Percentage step between strike prices 
    :return: DataFrame with strike prices and corresponding option prices 
    """ 
    strike_prices = [S0 * (1 + i / 100) for i in range(min_range, max_range + 1, step)] 
    option_prices = [black_scholes(S0, strike, T, r, sigma) for strike in strike_prices] 
 
    df = pd.DataFrame({
        'Strike Price': strike_prices, 
        'Option Price': option_prices 
    })
    

    return df
 
# Example usage
S0 = 100  # Close Price
r = 0.05  # Risk-free interest rate
T = 1     # Time to expiration (length of investment window) D / 365
sigma = 0.2  # forecasted daily volatility of the stock's returns
df = calculate_strike_prices(S0, r, T, sigma)
print(df)



    

