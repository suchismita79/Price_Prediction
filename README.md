# Price_Prediction
Prediction  with Deep Learning using Tensorslow
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 17:07:18 2018

import keras
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import time
from datetime import timedelta
start_time = time.monotonic()

path1="C:/Suchismita/DARPA/Platform Comparison/Data/House_Price_MOD2/Train/"
path2="C:/Suchismita/DARPA/Platform Comparison/Data/House_Price_MOD2/Test/"


res_DT=[]

############################################
def reg_model():
    # Initialising the ANN
    model = Sequential()
# Adding the input layer and the first hidden layer
    model.add(Dense(output_dim = 200, activation = 'relu', input_dim = 233, kernel_initializer='normal'))
# Adding the second hidden layer
    model.add(Dense(output_dim = 150, activation = 'relu'))
    model.add(Dense(output_dim = 100, activation = 'relu'))
    model.add(Dense(output_dim = 50, activation = 'relu'))
    model.add(Dense(output_dim = 25, activation = 'relu'))
    model.add(Dense(output_dim = 15, activation = 'relu'))

# Adding the output layer
    model.add(Dense(output_dim = 1, kernel_initializer='normal'))
# Compiling the ANN
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    return model

######################################################
x= os.listdir(path1)
y= os.listdir(path2)

# Declaring empty train and test list
train_list=[]
test_list=[]

seed =7
np.random.seed(seed)

# Creating separate list for train and test sets
for i,file in enumerate(x):
    train_list.append(file)
        
for i,file in enumerate(y):
    test_list.append(file)

for i in range(len(train_list)):
    dataset = pd.read_csv(path1+train_list[i])
    dataset1=pd.read_csv(path2+test_list[i])
    print ("Processed Train and Test Set"+str(i+1))
    print (train_list[i])
    
##############  House Price #############################

    dataset = dataset.drop(dataset.columns[0:2],axis =1)
    dataset1 = dataset1.drop(dataset1.columns[0:2],axis =1)
    
    dataset['Diff_Year']=dataset['YearRemodAdd'] - dataset['YearBuilt']
    dataset1['Diff_Year']=dataset1['YearRemodAdd'] - dataset1['YearBuilt']
    
    train_data= dataset.drop(['YearBuilt','YearRemodAdd','YrSold'],axis=1)
    test_data= dataset.drop(['YearBuilt','YearRemodAdd','YrSold'],axis=1)
    

    train_x=train_data.drop('SalePrice',1)
    train_y=train_data.loc[:,'SalePrice']
                    
    test_x=test_data.drop('SalePrice',1)
    test_y=test_data.loc[:,'SalePrice']

   
# Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    train_x = sc.fit_transform(train_x)
    test_x = sc.transform(test_x)


    estimator = KerasRegressor(build_fn=reg_model, epochs=150, batch_size=10, verbose=0)
    estimator.fit(train_x, train_y)
    #################  House Price Training End #######################################

for i in range(len(test_list)):
    dataset1=pd.read_csv(path2+test_list[i])
    print("Now Testing Files")
    y_pred1 = estimator.predict(test_x)

    from math import sqrt
    import sklearn.metrics
    from sklearn.metrics import mean_squared_error
    MSE1= mean_squared_error(train_y,y_pred1)
    RMSE1= sqrt(MSE1)
    res_DT.append(RMSE1)
end_time = time.monotonic()    
   
print("RMSE::",np.mean(res_DT),"Std Dev",np.std(res_DT))

print()

print("Total time: {}".format(timedelta(seconds=end_time - start_time)))    










