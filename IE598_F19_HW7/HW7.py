#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 17:16:59 2019

@author: howingshan
"""

import pandas as pd
df = pd.read_csv('/Users/howingshan/Desktop/2019FALL/IE598/HW6/cc.csv',header=None)
print(df.shape)
print(df.head())
summary = df.describe()
print(summary)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics

df=df.dropna()
df=df.drop(0)
X=df.drop(24,1).values
m=[24]
y=df[m].values
acc=[]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1, random_state=42)
#part1
from sklearn.ensemble import RandomForestClassifier
cvsm=[]
outacc=[]
a=np.array([10,20,30,40,50])
for i in range(0,5):
 forest=RandomForestClassifier(criterion='gini',n_estimators=a[i],random_state=1)
 forest.fit(X_train,y_train)
 y_pred=forest.predict(X_train)
 cvscores = cross_val_score(estimator=forest, X=X_train, y=y_train, cv=10, n_jobs=1)
 mean = np.mean(cvscores)
 cvsm.append(mean)
 y_test_pred = forest.predict(X_test)
 out_score = metrics.accuracy_score(y_test_pred, y_test)
 outacc.append(out_score)
 
for i in range(0,5):
   print("The mean cvscores of :",a[i],"essimators is ",cvsm[i])

for i in range(0,5):
   print("The out-sample's scores of :",a[i],"essimators is ",outacc[i])

 

#part2
feat_labels = df.columns[1:]
forest = RandomForestClassifier(n_estimators=500,
                                random_state=1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]
col=np.array(['LIMIT_BAL','SEX','EDUCATION','MARRIAGE',	'AGE','PAY_0','PAY_2','PAY_3',
     'PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4',
     'BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4',
     'PAY_AMT5','PAY_AMT6','DEFAULT'])  
for f in range(X_train.shape[1]):
    n=feat_labels[indices[f]]
    print("%2d) %-*s %f" % (f + 1, 20, 
                            col[n-1], 
                            importances[indices[f]]))
import matplotlib.pyplot as plt 
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), 
        importances[indices],
        align='center')

plt.xticks(range(X_train.shape[1]), 
           col[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

from sklearn.feature_selection import SelectFromModel

sfm = SelectFromModel(forest, threshold=0.05, prefit=True)
X_selected = sfm.transform(X_train)
print('Number of features that meet this threshold criterion:', 
      X_selected.shape[1])

for f in range(X_selected.shape[1]):
    m=feat_labels[indices[f]]
    print("%2d) %-*s %f" % (f + 1, 20, 
                            col[m-1], 
                            importances[indices[f]]))





print("My name is {Yingshan He}")
print("My NetID is: {yh29}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

