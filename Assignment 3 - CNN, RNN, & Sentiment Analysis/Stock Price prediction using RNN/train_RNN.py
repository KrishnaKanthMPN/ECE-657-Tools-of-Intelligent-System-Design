# import required packages

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np 
import pandas as pd 
import tensorflow as tf
from tensorflow import keras
from subprocess import check_output
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM,GRU,SimpleRNN
from keras.models import Sequential
from sklearn.model_selection import  train_test_split
import time 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt 
from numpy import newaxis
import math
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
import pickle
tf.random.set_seed(221)
np.random.seed(221)
os.environ['PYTHONHASHSEED']=str(221)

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow

#Dataset creation has been commented out as requested.
'''
# Reading and storing the "q2_dataset.py" given
stocks_dataset =  pd.read_csv('data/q2_dataset.csv', header=0,parse_dates=['Date'])
stocks_dataset.columns = ['Date', 'close','volume','open','high','low']
stocks_dataset['close'] = stocks_dataset['close'].replace('[\$,]', '', regex=True).astype(float)
stocks_dataset=stocks_dataset.sort_values(by='Date',ascending=False)
#stocks_dataset

df = pd.DataFrame(columns=['open1', 'high1', 'low1', 'volume1',
                           'open2', 'high2', 'low2', 'volume2',
                           'open3', 'high3', 'low3', 'volume3','target'])

l= len(stocks_dataset.index)
i=0
while i < l-3:

    df.loc[i, 'open1'] = stocks_dataset["open"][i+3]
    df.loc[i, 'high1'] = stocks_dataset["high"][i+3]
    df.loc[i, 'low1'] = stocks_dataset["low"][i+3]
    df.loc[i, 'volume1'] = stocks_dataset["volume"][i+3]
    
    df.loc[i, 'open2'] = stocks_dataset["open"][i+2]
    df.loc[i, 'high2'] = stocks_dataset["high"][i+2]
    df.loc[i, 'low2'] = stocks_dataset["low"][i+2]
    df.loc[i, 'volume2'] = stocks_dataset["volume"][i+2]
    
    df.loc[i, 'open3'] = stocks_dataset["open"][i+1]
    df.loc[i, 'high3'] = stocks_dataset["high"][i+1]
    df.loc[i, 'low3'] = stocks_dataset["low"][i+1]
    df.loc[i, 'volume3'] = stocks_dataset["volume"][i+1]    
    
    df.loc[i, 'target'] = stocks_dataset["open"][i]      

    i =i+1
    
df.to_csv('data/final.csv', mode='w+', header=True, index=False)
final_dataset =  pd.read_csv('data/final.csv', header=0)

train,test = train_test_split(final_dataset, test_size=0.30, random_state=42)
train.to_csv('data/train_data_RNN.csv', index=False, mode='w+')
test.to_csv('data/test_data_RNN.csv', index=False, mode='w+')
'''




if __name__ == "__main__": 
	# 1. load your training data
	train_dataset =  pd.read_csv('data/train_data_RNN.csv')
	#Data Normalization using MinMaxScaler
	y_train=train_dataset[['target']]
	x_train=train_dataset.drop(columns=['target'])

	MinMaxObject_x = MinMaxScaler(feature_range=(0,1))
	MinMaxObject_y = MinMaxScaler(feature_range=(0,1))
	norm_x_train = MinMaxObject_x.fit_transform(x_train)
	norm_y_train = MinMaxObject_y.fit_transform(y_train)

	#Saving the MinMax objects as pickle files as the same object is used by testRNN.py
	#https://stackoverflow.com/questions/50565937/how-to-normalize-the-train-and-test-data-using-minmaxscaler-sklearn/50567308
	pickle.dump(MinMaxObject_x, open("data/MinMaxObject_x.pkl", 'wb'))
	pickle.dump(MinMaxObject_y, open("data/MinMaxObject_y.pkl", 'wb'))
	
	# Splitting each normalized data sample into 3 arrays (Timestamps: t-1, t-2 and t-3)
	# This will convert the data as 3 timestamps data with 4 features in each timestamp
	dummy_x_train=[]
	for i in range(len(norm_x_train)):
		z=np.split(norm_x_train[i], 3)
		dummy_x_train.append(np.array(z))
	new_x_train=np.array(dummy_x_train)
	#new_x_train.shape

	# 2. Train your network
	# 		Make sure to print your training loss within training to show progress
	# 		Make sure you print the final training loss
	model_lstm_3 = Sequential()
	#adding 1st lstm layer
	model_lstm_3.add(LSTM(units = 200, return_sequences = True, input_shape = (new_x_train.shape[1], 4)))
	model_lstm_3.add(Dropout(rate = 0.2))

	##adding 2nd lstm layer
	model_lstm_3.add(LSTM(units = 200, return_sequences = False))
	model_lstm_3.add(Dropout(rate = 0.2))
	
	##adding output layer
	model_lstm_3.add(Dense(units = 1))

	model_lstm_3.compile(optimizer='adam', loss='mean_squared_error')

	model_lstm_3.fit(new_x_train, norm_y_train, epochs=1000, batch_size=64,verbose=1)

	# Model training loss
	train_predict = model_lstm_3.predict(new_x_train) 
	train_predict = MinMaxObject_y.inverse_transform(train_predict)
	loss_train = mean_squared_error(y_train,train_predict)
	print("Loss on the training data of LSTM Model 3 (MSE)= " +str(loss_train))

	# 3. Save your model
	model_lstm_3.save('models\Group_55_RNN_model.hdf5')
	