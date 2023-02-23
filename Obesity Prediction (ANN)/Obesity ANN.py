import pandas as pd
import numpy as np

#Data Preprocessing
from sklearn.preprocessing import LabelEncoder
dataframe = pd.read_csv('diabetes.sp18.csv')
le = LabelEncoder()
x = dataframe.iloc[:,8:12]
x_cat = pd.DataFrame(le.fit_transform(x.iloc[:, 1]))
x = x.drop('sex', axis =1)
x = np.concatenate((x, x_cat), axis = 1)
y = dataframe.iloc[:,13]
y = le.fit_transform(y)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])


#Split Data Set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:, :3] = sc.fit_transform(x_train[:, :3])
x_test[:, :3] = sc.transform(x_test[:, :3])

#Create ANN
from keras.models import Sequential
from keras.layers import Dense

ann = Sequential()
ann.add(Dense(units = 10, activation = 'relu'))
ann.add(Dense(units = 10, activation = 'relu'))
ann.add(Dense(units = 10, activation = 'relu'))
ann.add(Dense(units = 1, activation = 'sigmoid'))

ann.compile(optimizer = ('adam'), loss = ('binary_crossentropy'),metrics = ['accuracy'])
ann.fit(x_train,y_train, batch_size = 32, epochs = 50)

#Predict Values
y_pred = ann.predict(x_test)
y_pred = (y_pred > .5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

#Predict Single Value
one_p = [30, 60, 200, 0]
s_pred = ann.predict([one_p])
if s_pred > .5:
    s_pred = 'obese'
else:
    s_pred = 'not obese'
print (s_pred)

