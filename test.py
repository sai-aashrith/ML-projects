import pandas_datareader as pdr
import pandas as pd
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import math
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping
from keras.models import load_model

# Retrieving data
key = '9112e261ca3a7f736115d551117067134314897a'
df = pdr.get_data_tiingo('AAPL', api_key=key)
df.to_csv('AAPL.csv')
df = pd.read_csv('AAPL.csv')
df1 = df.reset_index()['close']

# Preprocessing data
scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

# Creating training and testing data
train_size = int(len(df1) * .75)
test_size = len(df1) - train_size
train_data, test_data = df1[0:train_size, :], df1[train_size:len(df1), :1]


def create_dataset(dataset, timestep):
    dataX, dataY = [], []
    for i in range(len(dataset) - 1 - timestep):
        a = dataset[i:(i + timestep), 0]
        dataX.append(a)
        b = dataset[i + timestep, 0]
        dataY.append(b)
    return np.array(dataX), np.array(dataY)


time_step = 200
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# # Model
model = Sequential()

# model.add(LSTM(50,return_sequences=True,input_shape=(200,1)))
# model.add(LSTM(50,return_sequences=True))
# model.add(LSTM(50))
# model.add(Dense(1))

model.add(LSTM(32, return_sequences=True, input_shape=(200, 1), recurrent_initializer='glorot_uniform'))
model.add(LSTM(64, return_sequences=True, recurrent_initializer='glorot_uniform'))
model.add(LSTM(64))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='val_loss', patience=3)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=75, batch_size=32, verbose=1,
          callbacks=[early_stop])
model.save('stock_pred.h5')

# # Predictions and performance metrics
train_predict = model.predict(X_train)

test_predict = model.predict(X_test)
# Perform inverse transformation
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
print('train RMSE = ', math.sqrt(mean_squared_error(y_train,train_predict)))
print('test RMSE = ', math.sqrt(mean_squared_error(y_test,test_predict)))
# print(y_train.shape, test_predict.shape)
#
# Plotting
look_back = 200
trainPredictPLot = np.empty_like(df1)
trainPredictPLot[:, :] = np.nan
trainPredictPLot[look_back:len(train_predict)+ look_back,:] = train_predict
testPredictPLot = np.empty_like(df1)
testPredictPLot[:, :] = np.nan
testPredictPLot[len(train_predict)+ (look_back*2)+1: len(df1)-1,:] = test_predict
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPLot)
plt.plot(testPredictPLot)
plt.show()

# Forecasting
x_input = test_data[-time_step:].reshape(1,-1)
temp_input = list(x_input)
temp_input=temp_input[0].tolist()
lst_output = []
n_steps = 200
i=0
while(i<31):
    if(len(temp_input)>200):
        x_input = np.array(temp_input[1:])
        print('{} day input{}'.format(i,x_input))
        x_input = x_input.reshape(1,n_steps,1)
        yhat = model.predict(x_input)
        print('{} day output{}'.format(i, yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape(1,n_steps,1)
        yhat = model.predict(x_input)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        i=i+1
print(lst_output)
day_new = np.arange(1,201)
day_pred = np.arange(201,231)
df2 = df1.tolist()
df2.extend(lst_output)
plt.plot(day_new,scaler.inverse_transform(df1[-200:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))
plt.show()


