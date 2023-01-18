"""
Ethereum and Bitcoin market traking and prediction using a LSTM

- I will loop primarily at the trends between closing prices for my data points

Historic-Crypto python package (pip3 install Historic-Crypto)
This Package works with Coinbase Pro API & will allow me to:

1. Return Historical data in the form of Pandas Dataframe
// This data is saved to csv files and code commented out to reduce execution time.
"""

# Importing Libraries 
#from Historic_Crypto import HistoricalData #the data source

"""
Writing data to csv files below
This data preprocessing step is taken to ensure that the API does not need to be called every time the program is executed.
This saves an enormous amount of execution time.
These are commented out because the data is saved in the csv files for future use
"""

# (Historical Data returned as dataframe) 
#eth_hist = HistoricalData('ETH-USD', 300, '2021-01-01-00-00').retrieve_data()
#btc_hist = HistoricalData('BTC-USD', 300, '2021-01-01-00-00').retrieve_data()

# Writing to CSV files
#eth_hist.to_csv('eth-hist.csv')
#btc_hist.to_csv('btc-hist.csv')





###  Program library imports  ###
from keras.callbacks import History
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.dates as dt
from datetime import datetime
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers.core import Activation, Dense


###  Reading data from CSV files  ###
eth_hist = pd.read_csv('eth-hist.csv')
btc_hist = pd.read_csv('btc-hist.csv')


# Function to show the history of Market close price for 2021
def show_plot(value, color):
    if value == 'eth-hist':
        v = eth_hist
    elif value == 'btc-hist':
        v = btc_hist
    ###  Creating a graph of BTC-History data  ###
    dates = [] #list of dates
    times = [] #list of times 
    xvalues = [] #list of formatted dates

    for i in range(0,len(v)):
        dates.append(str(v['time'][i][:10]))
        times.append(str(v['time'][i][11:]))
        xvalues.append(datetime(int(dates[i][:4]), int(dates[i][5:7]), int(dates[i][8:10])))

    titlestr = str(value) + ' 2021 Prices'
    plt.plot_date(dt.date2num(xvalues), v['close'], color)
    plt.title(titlestr)
    plt.xlabel('Date')
    plt.ylabel('Price (US Dollars)')
    plt.show()

# Displaying Market close price history for training data visualization
#show_plot('btc-hist', 'bo')
#show_plot('eth-hist', 'ro')



###  DATA PREPROCESSING STEPS  ###

# Function compresses data into ranges between 0,1
# this is a necessary step to help the optimizer extract features from the data better
es = MinMaxScaler(feature_range=(0,1))
bs = MinMaxScaler(feature_range=(0,1))
def compress_data(eth, btc):
    # Ethereum
    eclose = eth['close'].values.reshape(-1,1)
    eth_scaled_close = es.fit_transform(eclose)

    #Bitcoin
    bclose = btc['close'].values.reshape(-1,1)
    btc_scaled_close = bs.fit_transform(bclose)

    return eth_scaled_close, btc_scaled_close
    
esc, bsc = compress_data(eth_hist, btc_hist)

# Function to prepare the data into short bites for the LSTM model
# returns a numpy array to work with sklearn better
# Transforming compressed data into data bites
def data_bites(data):
    # Data will be prepared into 2 hour bites (24 data points of 5 minutes)
    bites = []
    for i in range(len(data) - 24):
        # appending 24 data points to each index in bites list
        bites.append(data[i:i+24])
    return np.array(bites) # returning a numpy array of the data bite sequences

e_bites = data_bites(esc)
b_bites = data_bites(bsc)


# Function to process the rest of the data and split to train/test
# returns a partitioned dataset of xtrain, ytrain, xtest, ytest
def process_data(bites):
    train_length = math.ceil(len(bites) * .8) #split data to train/test
    X_train = bites[:train_length, :-1, :] #first 80% of data values (not including output value)
    Y_train = bites[:train_length, -1, :] #first 80% of data values (only output value)
    X_test = bites[train_length:, :-1, :] #last 20% of data values (not including output value)
    Y_test = bites[train_length:, -1, :] #last 20% of data values (only output value)
    return X_train, X_test, Y_train, Y_test

eX_train, eX_test, eY_train, eY_test = process_data(e_bites)
bX_train, bX_test, bY_train, bY_test = process_data(b_bites)



###  Building the LSTM keras model  ###
###  LSTM expects 3D data [batch_size, sequence length, #features]
"""
Parameters & HyperParameters to be Otpimized:
1. epochs (1 through 10)
2. Batch Size (32, 64, 128)
"""

# Function to build the LSTM model 
# Output is a model which predicts future cryptocurrency prices
def build_LSTM():
    model = Sequential()

    model.add(Bidirectional(
        #was getting index out od range error on 24 fixed by changing to 23
        LSTM(23, return_sequences=True),
        input_shape=(23, 1)
    ))
    model.add(Dense(units=1))
    model.add(Activation('linear'))

    return model

# Ethereum and Bitcoin models
e_model = build_LSTM()
b_model = build_LSTM()

# Function to train the LSTM Ethereum model 
def e_train_model(model):
    batch = 128 # 32, 64, 128 tested and recorded
    epochs = 10 # 1-10 will be tracked
    model.compile(loss='mse', optimizer='Adamax')

    model.fit(
        #ethereum training data
        eX_train,
        eY_train,
        epochs = epochs, #epochs (hyperparameter)
        batch_size = batch, #batch size (hyperparameter)
        shuffle=False #order needed for time series forecasting
    )

# Function to train the LSTM Bitcoin model
def b_train_model(model):
    batch = 128 # 32, 64, 128 tested and recorded
    epochs = 10 # 1-10 will be tracked
    model.compile(loss='mse', optimizer='Adamax')

    model.fit(
        #bitcoin training data
        bX_train,
        bY_train,
        epochs = epochs, #epochs (hyperparameter)
        batch_size = batch, #batch size (hyperparameter)
        shuffle=False #order needed for time series forecasting
    )


# Training Ethereum and Bitcoin models
e_train_model(e_model)
b_train_model(b_model)

# Predicting test results after being trained
e_predict = e_model.predict(eX_test)
b_predict = b_model.predict(bX_test)
# reshaping predictions as a 2D array
e_predict = e_predict.reshape(e_predict.shape[0], e_predict.shape[1])
b_predict = b_predict.reshape(b_predict.shape[0], b_predict.shape[1])


# inverting ethereum prediction data back to original prices with inverse_transform
ey_test_inverse = es.inverse_transform(eY_test)
ey_predict_inverse = es.inverse_transform(e_predict)

# inverting bitcoin prediction data back to original prices with inverse_transform
by_test_inverse = bs.inverse_transform(bY_test)
by_predict_inverse = bs.inverse_transform(b_predict)


# Function to determine the model accuracy and loss value
def Results(model, testX, testY):
    res = model.evaluate(testX, testY)
    print(res)
print("Ethereum Results:\n")
Results(e_model, eX_test, eY_test)
print("Bitcoin results:\n")
Results(b_model, bX_test, bY_test)


# Function to plot the predictions
def plot_predictions(model, y_pred):
    if model == e_model:
        val = "Ethereum"
    elif model == b_model:
        val = "Bitcoin"
    plt.plot(y_pred, label="Model Prediction Prices", color='red')

    titlestr = val + " Predicted Price Chart"
    plt.title(titlestr)
    plt.xlabel('Time (5 minute increments)')
    plt.ylabel('Price (US Dollars)')
    plt.show()

def plot_actual(model, y_real):
    if model == e_model:
        val = "Ethereum"
    elif model == b_model:
        val = "Bitcoin"

    plt.plot(y_real, label="Real world Prices", color='green')
    titlestr = val + " Actual Price Chart"
    plt.title(titlestr)
    plt.xlabel('Time (2 hour increments)')
    plt.ylabel('Price (US Dollars)')
    plt.show()

plot_predictions(e_model, ey_predict_inverse)
plot_actual(e_model, ey_test_inverse)


plot_predictions(b_model, by_predict_inverse)
plot_actual(b_model, by_test_inverse)
