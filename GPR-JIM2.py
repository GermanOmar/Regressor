# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 20:07:26 2020
Algorithm: Gaussian process regression
@author: gobarrionuevo@uc.cl
"""

# Numpy (data import, manipulation, export)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import preprocessing

# load the data file
print('Gaussian process regression: ')
print()

df1=pd.read_table('DAT.txt',header=None)
x=df1.loc[:,0:4]
y=df1.loc[:,5]


#DATA SPLIT
from sklearn.model_selection import train_test_split #Divido los datos para entrenamiento y validacion
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
#KERNEL
GP = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)
#GP = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100, alpha=0.1, normalize_y=True)
#GP = GaussianProcessRegressor(kernel=None, alpha=1e-10, n_restarts_optimizer=0, random_state=None)

#START TRAINING
GP.fit(x_train,y_train)
#PREDICTION
y_pred=GP.predict(x_test)

#METRICS: R2, RMSE, MAPE
from sklearn.metrics import r2_score
R2=r2_score(y_test, y_pred)
RMSE = np.square(np.subtract(y_test,y_pred)).mean()
MAPE = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

IM=np.sqrt((R2*R2) + (RMSE*RMSE) + (MAPE*MAPE) )
print("IM: ", IM)

#PLOT model:
plt.scatter(y_test,y_pred)
plt.title('GP')
plt.xlabel('True')
plt.ylabel('Predicted')
plt.show()

print('THE END')

