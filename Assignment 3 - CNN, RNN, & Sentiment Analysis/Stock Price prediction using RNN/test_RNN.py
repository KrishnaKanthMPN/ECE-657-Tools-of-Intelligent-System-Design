# import required packages

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import math
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import pickle

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow



if __name__ == "__main__":
	# 1. Load your saved model
	model_lstm_3 = keras.models.load_model('models\Group_55_RNN_model.hdf5')

	# 2. Load your testing data
	# Reading the "test_data_RNN.csv" which is created above
	test_dataset =  pd.read_csv('data/test_data_RNN.csv')
	y_test=test_dataset[['target']]
	x_test=test_dataset.drop(columns=['target'])
	#y_test.shape

	#Reading the pickle file for MinMax Objects
	MinMaxObject_x = pickle.load(open("data/MinMaxObject_x.pkl", 'rb'))
	MinMaxObject_y = pickle.load(open("data/MinMaxObject_y.pkl", 'rb'))

	norm_x_test = MinMaxObject_x.transform(x_test) #MinMaxScaler
	norm_y_test = MinMaxObject_y.transform(y_test)

	#Data Reshape
	#Splitting the data into arrays - each sample has 3 timestamps of 4 features each
	#This step is carried out so as to provide the input to the LSTM model appropriately
	dummy_x_test=[]
	for i in range(len(norm_x_test)):
		k=np.split(norm_x_test[i], 3)
		dummy_x_test.append(np.array(k))
		
	new_x_test=np.array(dummy_x_test)
	#new_x_test.shape

	# 3. Run prediction on the test data and output required plot and loss
	value_predicted = model_lstm_3.predict(new_x_test)
	test_predict = MinMaxObject_y.inverse_transform(value_predicted)
	#rmse_test = math.sqrt(mean_squared_error(y_test,test_predict))
	loss_test = mean_squared_error(y_test,test_predict)
	print("The performance metric considered is Mean squared Error")
	print("Loss on the testing data of LSTM Model 3 (MSE)= " +str(loss_test)) 

	#Plot
	plt.figure(figsize=(14, 10))
	plt.plot(y_test.values[::-1], linestyle='--',color='yellow',marker='o')
	plt.plot(test_predict[::-1],linestyle='--', color= 'green',marker='*')
	plt.legend(['True Value','Predicted Value'], loc='upper right')
	plt.title("Opening price Prediction")
	plt.xlabel("Samples")
	plt.ylabel("Stocks Opening Price")
	plt.show()