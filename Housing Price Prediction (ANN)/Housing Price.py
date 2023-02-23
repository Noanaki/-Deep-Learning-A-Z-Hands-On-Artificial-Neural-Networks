import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import sklearn as sk

dataset = pd.read_csv('Housing.csv')
x = dataset.iloc[:, 1:]
y = dataset.iloc[:, 0]

x = x.to_numpy()

x[:, 5:11] = (x[:,5:11]=='yes').astype(float)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [11])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))

x = np.asarray(x).astype(np.float32)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .25, random_state = 0)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
x_train[:, :5] = sc.fit_transform(x_train[:, :5])
x_test[:, :5]= sc.transform(x_test[:, :5])


from keras.models import Sequential
from keras.layers import Dense

ann = Sequential()
ann.add(Dense(units=50, activation = 'relu'))
ann.add(Dense(units=50, activation = 'relu'))
ann.add(Dense(units=50, activation = 'relu'))
ann.add(Dense(units=50, activation = 'relu')) 
ann.add(Dense(units=50, activation = 'relu'))
ann.add(Dense(units=1, activation = 'linear'))

ann.compile(optimizer=('adam'), loss=('mean_squared_error'))
ann.fit(x_train, y_train, batch_size=(32), epochs=100)


