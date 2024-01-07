import math
from datetime import datetime, timezone
import pandas as pd
from numberstwo import *

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
    