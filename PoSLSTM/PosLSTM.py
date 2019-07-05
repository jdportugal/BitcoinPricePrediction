# -*- coding: utf-8 -*-
from keras.layers.core import Activation, Dense, Dropout, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras import optimizers
import numpy as np
from keras.callbacks import CSVLogger
from keras.utils import plot_model
from keras.models import load_model
import random

INPUT_PATH = "BTCUSD_1hr.csv"
TIME_STEPS = 50
BATCH_SIZE = 100

random.seed(5320)

def build_timeseries(mat, y_col_index):
    # y_col_index is the index of column that would act as output column
    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - TIME_STEPS
    print("-----\n")
    print(dim_0)
    print("-----\n")
    dim_1 = mat.shape[1]
    print("-----\n")
    print(dim_1)
    print("-----\n")
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,2))
    for i in range(dim_0):
        x[i] = mat[i:TIME_STEPS+i]
        y[i] = mat[TIME_STEPS+i:TIME_STEPS+i+2, y_col_index]
        #print(y[i])
    print("length of time-series i/o",x.shape,y.shape)
    return x, y

def trim_dataset(mat, batch_size):
    """
    trims dataset to a size that's divisible by BATCH_SIZE
    """
    no_of_rows_drop = mat.shape[0]%batch_size
    if(no_of_rows_drop > 0):
        return mat[:-no_of_rows_drop]
    else:
        return mat

df_ge = pd.read_csv('conll_tagged_text_allV2.txt', sep=" ", header=None)
#df_ge.sort_values(by=['Unix Timestamp'],inplace=True)
df_ge2 = df_ge
#print(df_ge2)
df_ge2.tail()
#print(df_ge)
'''
plt.figure()
plt.plot(df_ge2["Open"])
plt.plot(df_ge2["High"])
plt.plot(df_ge2["Low"])
plt.plot(df_ge2["Close"])
plt.title('GE stock price history')
plt.ylabel('Price (USD)')
plt.xlabel('Date')
plt.legend(['Open','High','Low','Close'], loc='upper left')
plt.show()

print(df_ge2.head())
print("checking if any null values are present\n", df_ge.isna().sum())

plt.figure()
plt.plot(df_ge2["Volume"])
plt.title('GE stock volume history')
plt.ylabel('Volume')
plt.xlabel('Days')
plt.show
'''

train_cols = [0,1]
df_train, df_test = train_test_split(df_ge, train_size=0.8, test_size=0.2, shuffle=False)
print("Train and Test size", len(df_train), len(df_test))

#Feature scaling
# scale the feature MinMax, build array
x = df_train.loc[:,train_cols].values
min_max_scaler = MinMaxScaler()
min_max_scaler_y = MinMaxScaler()
x_train = min_max_scaler.fit_transform(x)
x_test = min_max_scaler.transform(df_test.loc[:,train_cols])

y_train = min_max_scaler_y.fit_transform(df_test.loc[:,["Close"]].values)
y_temp = min_max_scaler_y.transform(df_test.loc[:,["Close"]])



x_t, y_t = build_timeseries(x_train, 1)
x_t = trim_dataset(x_t, BATCH_SIZE)
y_t = trim_dataset(y_t, BATCH_SIZE)


x_temp, y_temp = build_timeseries(x_test, 1)
x_val, x_test_t = np.split(trim_dataset(x_temp, BATCH_SIZE),2)
y_val, y_test_t = np.split(trim_dataset(y_temp, BATCH_SIZE),2)

'''
lstm_model = Sequential()
lstm_model.add(LSTM(100, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]), dropout=0.0, recurrent_dropout=0.0, stateful=False,   kernel_initializer='random_uniform'))
lstm_model.add(Dropout(0.5))
lstm_model.add(Dense(25,activation='relu'))
lstm_model.add(Dense(2,activation='sigmoid'))
optimizer = optimizers.RMSprop(lr=0.003)
lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)
print(lstm_model.summary())
#lstm_model.describe()
csv_logger = CSVLogger(('log' + '.log'), append=True)



history = lstm_model.fit(x_t, y_t, epochs=10, verbose=2, batch_size=BATCH_SIZE,
                    shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                    trim_dataset(y_val, BATCH_SIZE)), callbacks=[csv_logger])

lstm_model.save('testes.h5') 
'''
lstm_model = load_model('testes.h5') 
gotten = lstm_model.predict(x_test_t[-500:],batch_size=BATCH_SIZE)
print(gotten)
gotten = min_max_scaler_y.inverse_transform(gotten)
print(y_test_t[-500:])
real =min_max_scaler_y.inverse_transform(y_test_t[-500:].reshape(-1,1))
plt.figure()

plt.plot(gotten[-10:,0], color="g", label="gotten")
plt.plot(real[-10:,0], color="b", label="Real")
plt.show()