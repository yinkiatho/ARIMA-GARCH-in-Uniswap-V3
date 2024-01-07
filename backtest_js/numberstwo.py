import math

def round_number(number, decimal_places):
    factor_of_ten = 10 ** decimal_places
    return round(number * factor_of_ten) / factor_of_ten

def sum_array(arr):
    return sum(arr)

def parse_price(price, percent):
    rounder = 2 if percent else 4

    if price == 0:
        return 0
    elif price > 1000000:
        return int(price)
    elif price > 1:
        return round_number(price, 2)
    else:
        m = -math.floor(math.log(abs(price)) / math.log(10) + 1)
        return round_number(price, m + rounder)

def log_with_base(y, x):
    return math.log(y) / math.log(x)

def get_min(data, col):
    return min(p[col] for p in data)

def get_max(data, col):
    return max(p[col] for p in data)

def max_in_array(data, col):
    max_val = 0
    for d in data:
        d_max = get_max(d, col)
        max_val = d_max if d_max > max_val else max_val
    return max_val

def min_in_array(data, col):
    min_val = 0
    for d in data:
        d_min = round(get_min(d, col), 2)
        min_val = d_min if d_min < min_val else min_val
    return min_val

def gen_rand_between(min_val, max_val, decimal_places):
    rand = math.random() * (max_val - min_val) + min_val
    power = 10 ** decimal_places
    return math.floor(rand * power) / power

def format_large_number(n):
    abbrev = 'kmb'
    base_suff = math.floor(math.log(abs(n)) / math.log(1000))
    suffix = abbrev[min(2, base_suff - 1)]
    base = abbrev.index(suffix) + 1
    return round_number(n / 10 ** (3 * base), 2) + suffix if suffix else str(round_number(n, 2))

def round_with_factor(number):
    factor = len(str(int(number))) - 1
    return 10 ** factor