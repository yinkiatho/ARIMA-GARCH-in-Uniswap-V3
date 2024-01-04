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
from sklearn import metrics 
import defi.defi_tools as dft
import math 
import random
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
from ARIMA_GARCH import ARIMA_GARCH
from Backtester import Backtester


class Simulator():
    
    def __init__(self, Address='0xcbcdf9626bc03e24f779434178a73a0b4bad62ed', network=1):
        
        self.Address = Address
        self.network = network
        self.backtester = Backtester(Address, network)
        self.arima_garch = ARIMA_GARCH((1,1,1), (1,1))
        
    
    def simulate(windows=1, risk_params=0.95):
        
        return