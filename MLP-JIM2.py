# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 20:07:26 2020
Algorithm: Multi-layer Perceptron
@author: gobarrionuevo@uc.cl
"""

# Numpy (data import, manipulation, export)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

# load the data file
print('Multi-layer Perceptron: ')
print()

df1=pd.read_table('DAT.txt',header=None)
x=df1.loc[:,0:4]
y=df1.loc[:,5]
Xs = preprocessing.scale(x)

#DATA SPLIT
from sklearn.model_selection import train_test_split #Divido los datos para entrenamiento y validacion
x_train, x_test, y_train, y_test = train_test_split(Xs,y,test_size=0.2)

from sklearn.neural_network import MLPRegressor
#KERNEL
MLP = MLPRegressor(hidden_layer_sizes=2,solver='lbfgs',learning_rate='adaptive',activation='relu',random_state=1, max_iter=500)
#MLP = MLPRegressor(hidden_layer_sizes=(1000,1000,1000), activation='relu', solver='adam', max_iter=800)

#START TRAINING
MLP.fit(x_train,y_train)
#PREDICTION
y_pred=MLP.predict(x_test)

#METRICS: R2, RMSE, MAPE
from sklearn.metrics import r2_score
R2=r2_score(y_test, y_pred)
RMSE = np.square(np.subtract(y_test,y_pred)).mean()
MAPE = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

IM=np.sqrt((R2*R2) + (RMSE*RMSE) + (MAPE*MAPE) )
print("IM: ", IM)

#PLOT model:
plt.scatter(y_test,y_pred)
plt.title('MLP')
plt.xlabel('True')
plt.ylabel('Predicted')
plt.show()

print('THE END')