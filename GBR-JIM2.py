# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 20:07:26 2020
Algorithm: Gradient Boosting Regressor
@author: gobarrionuevo@uc.cl
"""

# Numpy (data import, manipulation, export)
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt 

# load the data file
print('Gradient Boosting Regressor: ')
print()

df1=pd.read_table('DAT.txt',header=None)
x=df1.loc[:,0:4]
y=df1.loc[:,5]
Xs = preprocessing.scale(x)

#DATA SPLIT
from sklearn.model_selection import train_test_split #Divido los datos para entrenamiento y validacion
x_train, x_test, y_train, y_test = train_test_split(Xs,y,test_size=0.2)

from sklearn.ensemble import GradientBoostingRegressor
#KERNEL
#GBR=GradientBoostingRegressor(n_estimators=100, random_state=0)
GBR=GradientBoostingRegressor(n_estimators=100, random_state=0, learning_rate=0.1, )

#START TRAINING
GBR.fit(x_train,y_train)
#PREDICTION
y_pred=GBR.predict(x_test)

#METRICS: R2, RMSE, MAPE
from sklearn.metrics import r2_score
from sklearn.metrics import r2_score
R2=r2_score(y_test, y_pred)
RMSE = np.square(np.subtract(y_test,y_pred)).mean()
MAPE = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

IM=np.sqrt((R2*R2) + (RMSE*RMSE) + (MAPE*MAPE) )
print("IM: ", IM)

#PLOT model:
plt.scatter(y_test,y_pred)
plt.title('GBR')
plt.xlabel('True')
plt.ylabel('Predicted')
plt.show()

print('THE END')