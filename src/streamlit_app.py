import streamlit.components.v1 as components
from datetime import datetime
import os
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards
import matplotlib.pyplot as plt
import webbrowser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
#from Backtester import Backtester
from simulator import Simulator
from utils import *


os.chdir('/Users/yinki/OneDrive/Python/Crypto Whales/src')
print(os.getcwd())

Pool_address = '0xcbcdf9626bc03e24f779434178a73a0b4bad62ed' # WETH/WBTC pool 0.3% fee
simulator = Simulator(Address='0xcbcdf9626bc03e24f779434178a73a0b4bad62ed')


st.set_page_config(
    page_title="Crypto Whales 🐳",
    page_icon="🧊",
    layout="wide",
    # Add theme to be light
    initial_sidebar_state="expanded"
)

start_date = "2023-05-25"
end_date = "2023-12-24"
df = pd.read_csv('../data/pools_daily_weth_btc_arima_garch.csv', index_col=0, parse_dates=True,sep=';').loc[start_date:end_date]

st.title('Crypto Whales Streamlit App')
st.set_option('deprecation.showPyplotGlobalUse', False)


# Side Bar
st.sidebar.header("Model Configuration")


with st.sidebar:
    # General Analysis 
    risk_param = st.slider("Confidence Interval for Boundaries", 0.0, 1.0, step=0.01, value=0.95)
    # Choose Number of Stocks
    initial_investment = st.slider("Initial Investment", 1000000, 2000000, step=1000)
    
    # Choose Prediction Window
    windows = st.slider("Number of Windows", 1, 7, step=1, value=5)
    
    
st.header("Project Description")
st.write("Simulation Engine based on Crypto Whales: Hold On Tight Strategy")
st.subheader("Looking at our Test Period")
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Predicted Close (WBTC)'], label='Predicted Close', marker='o')
plt.plot(df.index, df['Close (WBTC)'], label='Actual Close', marker='o')
plt.fill_between(df.index, df['Predicted Close (WBTC)'] - 1.96 * df['Conditional Volatility'], df['Predicted Close (WBTC)'] + 1.96 * df['Conditional Volatility'], color='gray', alpha=0.2, label='Confidence Interval (95%)')
plt.title('WBTC Predicted Close with Confidence Interval')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
st.pyplot()


with st.spinner("Simulating LP Position..."):
    results = simulator.simulate(windows, risk_param, initial_investment)
    

st.header("Simulation Results")



# Plot the windows
# Plotting
start_dates = results['Start Date']
end_dates = results['End Date']
start_prices = results['Start Price (WBTC)']
end_prices = results['End Price (WBTC)']
lower_bounds = results['Lower Bound']
upper_bounds = results['Upper Bound']
df.index = pd.to_datetime(df.index)

cmap = cm.get_cmap('viridis')

plt.figure(figsize=(15, 6))
plt.plot(df.index, df['Close (WBTC)'], label='Predicted Close', marker='o')
plt.title('Boundaries of Pool Close with Confidence Interval')
plt.xlabel('Date')
plt.ylabel('Close Price')
# Plotting each window with its boundaries
for i, (start_date, end_date, start_price, end_price, lower_bound, upper_bound) in enumerate(zip(start_dates, end_dates, start_prices, end_prices, lower_bounds, upper_bounds)):
    print(f'Window {i + 1}')
    #print(f'Start Date: {start_date}')
    #print(f'End Date: {end_date}')
    #print(f'Start Price: {start_price}')
    #print(f'End Price: {end_price}')
    window = df[(df.index >= start_date) & (df.index <= end_date)]
    #print(window)
    color = cmap(i / windows)
    # Convert DatetimeIndex to list of strings for plotting dotted line
    date_strings = [str(date) for date in window.index]

    # Plot the start and end prices as a dotted line
    plt.plot(window.index[[0, -1]], [start_price, end_price], linestyle='--', label=f'Estimated Trend Line {i + 1}')

    # Plot the lower and upper bounds
    plt.fill_between(window.index, lower_bound, upper_bound, color=color, alpha=0.2, label=f'Bounds {i + 1}')

st.pyplot()



window = df.loc[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]

fee_col1, fee_col2 = st.columns(2)
# Fees Columns
with fee_col1:
    plt.figure(figsize=(12, 6))
    plt.plot(results['Fee USD'], label='Fees', marker='o')
    plt.title('Fees')
    plt.xlabel('Intervals')
    plt.ylabel('Fees')
    plt.legend()
    plt.grid(True)
    st.pyplot()
    
with fee_col2:
    # Pie Chart of Fees Earned, dont really say much
    fees = results['Fee USD']
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(fees, labels=[f'Interval {i+1}' for i in range(len(fees))], autopct='%1.1f%%', startangle=90)
    ax.set_title('Distribution of Fees')
    st.pyplot()
    

    
apr_col1, apr_col2 = st.columns(2)
with apr_col1:
# Plotting APR Strategy vs APR Unbounded Bar Graphs

    # Generate x-axis values
    intervals = np.arange(1, windows + 1)
    colors = np.random.rand(len(intervals), 3)
    # Plotting APR Strategy
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.35
    bar_positions_strategy = intervals 
    for i, (position, apr) in enumerate(zip(bar_positions_strategy, results['APR Strategy'])):
        ax.bar(position, apr, bar_width, label=f'APR Strategy {i+1}', alpha=0.8, color=colors[i])
    #ax.bar(bar_positions_unbounded, results['APR Unbounded'], bar_width, label='APR Unbounded', alpha=0.8)
    ax.set_xlabel('Intervals')
    ax.set_ylabel('APR')
    ax.set_title('APR Strategy vs APR Unbounded')
    ax.set_xticks(intervals)
    ax.legend()
    st.pyplot()

with apr_col2:

    # Table of APRs
    apr_results = pd.DataFrame({
        'APR Strategy': results['APR Strategy'],
        'APR Unbounded': results['APR Unbounded']
    }, index=intervals)
    st.table(apr_results)


# Hedging Costs and Cumulative Investment
inv_col1, inv_col2 = st.columns(2)



with inv_col1:
# Data
    hedging_costs = results['Hedging Costs']
    payoff = results['Payoff']
    print(payoff)
    fees_usd = results['Fee USD']

    # Number of sets
    num_sets = len(hedging_costs)

# Set up the index for each set
    index = np.arange(num_sets)

    # Bar width
    bar_width = 0.3

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    bar1 = ax.bar(index, hedging_costs, bar_width, label='Hedging Costs')
    bar2 = ax.bar(index + bar_width, payoff, bar_width, label='Payoff', bottom=fees_usd)
    bar3 = ax.bar(index + bar_width, fees_usd, bar_width, label='Fees USD')

# Adding labels and title
    ax.set_xlabel('Index')
    ax.set_ylabel('Values')
    ax.set_title('Side-by-Side Bar Plot with Stacked Bars')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels([f'Set {i+1}' for i in range(num_sets)])
    ax.legend()

    st.pyplot()
    
    
with inv_col2:
    # Plot Mean Percentage of Active Liquidity
    plt.figure(figsize=(12, 6))
    plt.plot(results['Mean Percentage of Active Liquidity'], label='Mean Percentage of Active Liquidity', marker='o')
    plt.title('Mean Percentage of Active Liquidity')
    plt.xlabel('Intervals')
    st.pyplot()



cum_col1, cum_col2 = st.columns(2)
with cum_col1:

# Plot Cumulative Investment USD vs HODL 50-50
# Plotting
    num_intervals = [i for i in range(0, windows + 1)]
    cum_investment = [initial_investment] + results['Cumulative Investment USD']
    hodl = [initial_investment] + results['HODL 50-50']
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(num_intervals, cum_investment, label='Cumulative Investment USD', marker='o')
    ax.plot(num_intervals, hodl, label='Cumulative Investment HODL 50-50', marker='o')

    ax.set_xlabel('Intervals')
    ax.set_ylabel('Cumulative Investment (USD)')
    ax.set_title('Cumulative Investment Comparison')
    ax.legend()

    st.pyplot()

# Show Table of Cumulative Investment USD vs HODL 50-50
    cumulative_results = pd.DataFrame({
        'Cumulative Investment USD': results['Cumulative Investment USD'],
        'HODL 50-50': results['HODL 50-50']
    }, index=intervals)
    
    cumulative_results['Excess Return'] = (cumulative_results['Cumulative Investment USD'] - cumulative_results['HODL 50-50'])
with cum_col2:
    cumulative_results.plot.bar(figsize=(12, 6), rot=0, title='Cumulative Investment Comparison')
    st.pyplot()
    

st.table(cumulative_results)





IL = results['Impermanent Loss']
#x_values = [i for i in range(1, windows + 1)]
#x_values = [i for i in range(sum([len(i) for i in IL]))]
# Plotting
fig, axs = plt.subplots(nrows=len(IL), figsize=(12, 6))
colors = cm.viridis(range(len(IL)))

for i in range(len(IL)):
    axs[i].plot(IL[i], color=colors[i])
    axs[i].set_xlabel('Time Step')
    axs[i].set_ylabel('Impermanent Loss')
    axs[i].set_title(f'Interval {i + 1}')

plt.tight_layout()
plt.show()

plt.tight_layout()
st.pyplot()

IL_Statistics = []
for i in range(len(IL)):
    IL_Statistics.append([np.mean(IL[i]), np.std(IL[i]), np.min(IL[i]), np.max(IL[i])])

IL_Statistics = pd.DataFrame(IL_Statistics, columns=['Mean', 'Standard Deviation', 'Minimum', 'Maximum'], index=[i for i in range(1, windows + 1)])
st.table(IL_Statistics)






