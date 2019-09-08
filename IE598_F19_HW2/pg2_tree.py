#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 21:34:47 2019

@author: howingshan
"""

import pandas as pd
df = pd.read_csv('/Users/howingshan/Desktop/2019FALL/IE598/HW2/DS1.csv',header=None)

from sklearn.model_selection import train_test_split
import sklearn
print( 'The scikit learn version is {}.'.format(sklearn.__version__))

import numpy as np
df=df.drop(columns=0)
df=df.drop(columns=1)

df=np.array(df)
X, y = df[1:,0:8],df[1:,9]
print( X.shape, y.shape)
print( X[0], y[0])

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=33)
print( X_train.shape, y_train.shape)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
SEED=1
dt = DecisionTreeClassifier(max_depth=6, random_state=SEED)
dt.fit(X_train, y_train)
y_pred=dt.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Test set accuracy: {:.2f}".format(acc))


print("My name is {Yingshan He}")
print("My NetID is: {yh29}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

