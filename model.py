import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import json
import requests

df = pd.read_csv('Salary_Data.csv')

X = df.drop(columns = 'Salary',axis = 1)
Y = df['Salary']

X_train, X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

regressor = LinearRegression()
regressor.fit(X_train,Y_train)
y_pred = regressor.predict(X_test)

pickle.dump(regressor,open('model.pkl','wb'))

# loading model to compare the results

# model = pickle.load(open('model.pkl','rb'))
# print(model.predict([[1.8]]))



