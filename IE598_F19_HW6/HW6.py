#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 22:15:24 2019

@author: howingshan
"""

import pandas as pd
df = pd.read_csv('/Users/howingshan/Desktop/2019FALL/IE598/HW6/cc.csv',header=None)
print(df.shape)
print(df.head())
summary = df.describe()
print(summary)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score  
import numpy as np

df=df.dropna()
df=df.drop(0)
X=df.drop(24,1).values
m=[24]
y=df[m].values
acc=[]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1, random_state=42)

#part 1
time_start=time.time()
for i in range(1,11):
  tree = DecisionTreeClassifier(max_depth=6, random_state=i)
  tree.fit(X_train, y_train)
  y_pred=tree.predict(X_test)
  acc.append(accuracy_score(y_test, y_pred))
time_end=time.time()
print('Decision tree totally costs ',time_end-time_start,'s')
print(acc)

print("The mean of the scores is:",np.mean(acc))
print("The variance of the scores is:",np.var(acc))

#part2
from sklearn.model_selection import cross_val_score

cv_scores=[]
cv_mean=[]
cv_var=[]
cv_std=[]
time_start=time.time()
for i in range(1,11):
  tree = DecisionTreeClassifier(max_depth=6, random_state=i)
  a=cross_val_score(tree,X,y,cv=10)
  cv_scores.append(a)
  cv_mean.append(np.mean(a))
  cv_var.append(np.var(a))
  cv_std.append(np.std(a,ddof=1))
time_end=time.time()
print('Cross validation totally costs ',time_end-time_start,'s')

print(cv_scores)

print("The mean of the  CV scores is:",cv_mean)
print("The standard deviation of the CV scores is:",cv_std)

print(np.mean(cv_mean))
print(np.std(cv_scores))












print("My name is {Yingshan He}")
print("My NetID is: {yh29}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

