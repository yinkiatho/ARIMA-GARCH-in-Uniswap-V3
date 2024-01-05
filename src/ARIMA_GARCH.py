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


class ARIMA_GARCH():
    
    def __init__(self, arima_order, garch_order, testset = None, trainset = None):
        self.arima_order = arima_order
        self.garch_order = garch_order
        
        if testset is None:
            self.testset = pd.read_csv('data/test_returns.csv', index_col = 0)
        else:
            self.testset = testset
            
            
        if trainset is None:
            self.trainset = pd.read_csv('data/df_returns.csv', index_col = 0)
        else:
            self.trainset = trainset
        

        self.arima = ARIMA(self.trainset, order = self.arima_order).fit()
        self.garch = arch_model(self.arima.resid, p = self.garch_order[0], q = self.garch_order[1]).fit()
            
        
    def grid_search_optimize():
        return
    
    
    def fit_arima(self, new_arima_order):
        self.arima_order = new_arima_order
        self.arima = ARIMA(self.df, order = self.arima_order).fit()
        return
    
    
    def generate_predictions(self, start, end):
        predictions = self.arima.predict(start = start, end = end)
        return predictions
    
    
    def generate_volatilities(self, horizon=1):
        return 
    
    def generate_next_volatility(self):
        return self.garch.forecast(horizon = 1, reindex = False).variance.iloc[0, 0] ** 0.5
    
    def generate_next_prediction(self):
        return self.arima.forecast(steps = 1).values[0]
    
    
    def monte_carlo_sims(self, num_simulations):
        
        time_horizon = len(self.testset) - 1  # Adjust the time horizon based on the dataset size
        simulations = np.zeros((num_simulations, time_horizon + 1))
        start_price = self.trainset.iloc[-1, 0]
        simulations[:, 0] = start_price
        #print(simulations)
        #dataset = dataset.values
        test_dates = self.testset.index
        s = 0
        while s < num_simulations:
            np.random.seed(random.randint(0, 10000))
            current_dataset = self.trainset.copy()
            predictions = []
            predictions.append(start_price)

            for i in range(1, time_horizon + 1):
                # Use the observed data up to the current point for training
                train_data = current_dataset
            
                # Fit ARIMA model
                arima_model = sm.tsa.ARIMA(train_data, order=self.arima_order)
                arima_fit = arima_model.fit()

                # Generate GARCH innovations
                garch_innovations = arima_fit.resid

                # Fit GARCH model
                garch_model = arch_model(garch_innovations, vol='Garch', p=self.garch_order[0], q=self.garch_order[1])
                garch_fit = garch_model.fit(disp='off', show_warning=False)

                # Make a one-step forecast using the ARIMA-GARCH model
                arima_forecast = arima_fit.forecast(steps=1).values[0]
                garch_volatility = garch_fit.conditional_volatility[-1]
            
                #print(f'Predicted Close (WBTC): {arima_forecast}, Predicted Volatility: {garch_volatility}')
            
                #print(simulations[s, i- 1])
                path = arima_forecast + garch_volatility * np.random.normal(0, 1)
            
                # Update the dataset
                new_row = pd.Series([path], index=[test_dates[i]])
                current_dataset = current_dataset.append(new_row)
                predictions.append(path)
            
            simulations[s, :] = predictions
            s += 1
            print(f"Simulation {s} complete")

        return simulations
        
    
        
        
        