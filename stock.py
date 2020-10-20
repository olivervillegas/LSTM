import tensorflow as tf
import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import math
from sklearn.metrics import mean_squared_error
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
scaler = MinMaxScaler(feature_range=(0,1))

df1 = []
train_data = []
test_data = []
X_train = []
X_test = []
y_train = []
ytest = []
model = Sequential()

class stock:
    # Initialize a stock object
    def __init__(self, file_name, num_epochs, title, date):
        self.file_name = file_name
        self.num_epochs = num_epochs
        self.title = title
        self.date = date

    def train_and_graph(self):
        # Read values into a dataframe
        df = pd.read_csv(self.file_name)
        df1 = df.reset_index()['Close']
        df1 = df1.iloc[::-1]
        df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))
        training_size = int(len(df1) * 0.65)
        test_size = len(df1) - training_size
        train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1]

        time_step = 100
        # Create a dataset and store in array
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, ytest = create_dataset(test_data, time_step)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Layout the LSTM model
        model.add(LSTM(50, return_sequences=True, input_shape=(100,1)))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))

        # Compile and fit the model
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train, y_train, validation_data=(X_test,ytest),epochs=self.num_epochs, batch_size=64, verbose=1)

        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)

        math.sqrt(mean_squared_error(y_train, train_predict))
        math.sqrt(mean_squared_error(ytest, test_predict))

        look_back = 100
        trainPredictPlot = np.empty_like(df1)
        trainPredictPlot[:,:] = np.nan
        trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

        testPredictPlot = np.empty_like(df1)
        testPredictPlot[:,:] = np.nan
        testPredictPlot[len(train_predict)+ (look_back*2)+1:len(df1)-1,:] = test_predict

        plt.plot(scaler.inverse_transform(df1))
        plt.plot(trainPredictPlot)
        plt.plot(testPredictPlot)
        plt.xlabel('Business Days Since ' + self.date)
        plt.ylabel("Stock Market Price in USD")
        plt.title(self.title)
        plt.show()

        tdM100 = len(test_data) - 100
        x_input = test_data[tdM100:].reshape(1,-1)

        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()

        lst_output = []
        n_steps = 100
        i = 0

        while(i < 30):
            if(len(temp_input)>100):
                x_input = np.array(temp_input[1:])
                print("{} day input".format(i, x_input))
                x_input = x_input.reshape(1,-1)
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                lst_output.extend(yhat.tolist())
                i = i+1
            else:
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose = 0)
                print(yhat[0])
                temp_input.extend(yhat[0].tolist())
                print(len(temp_input))
                lst_output.extend(yhat.tolist())
                i = i+1

        print(lst_output)

        day_new = np.arange(1, 101)
        day_pred = np.arange(101, 131)

        print(len(df1))

        df3 = df1.tolist()
        df3.extend(lst_output)

        minus100 = len(df1) - 100

        plt.plot(day_new, scaler.inverse_transform(df1[minus100:]))
        plt.plot(day_pred, scaler.inverse_transform(lst_output))
        plt.xlabel('Business Days Since 7 April 2020')
        plt.ylabel("Stock Market Price in USD")
        plt.title(self.title)
        plt.show()


def create_dataset(dataset, timestep = 1):
    dataX, dataY = [], []
    for i in range(len(dataset)-timestep-1):
        a = dataset[i:(i+timestep),0]
        dataX.append(a)
        dataY.append(dataset[i + timestep,0])
    return np.array(dataX), np.array(dataY)