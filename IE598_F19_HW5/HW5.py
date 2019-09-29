#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 21:39:51 2019

@author: howingshan
"""

import sys
import csv
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('/Users/howingshan/Desktop/2019FALL/IE598/HW5/hw5.csv')
print(df.shape)
print(df.head())
summary = df.describe()
print(summary)
columns=df.columns[1:]



#split data
from sklearn.model_selection import train_test_split
df=df.dropna()
X=df.drop('Adj_Close',1)
X=X.drop('Date',1).values
m=['Adj_Close']
y=df[m].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state=42)
#standardalize
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
#eigenvalues
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

print('\nEigenvalues \n%s' % eigen_vals)

tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

x=np.linspace(1,30,30)
import matplotlib.pyplot as plt
plt.bar(x, var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(x, cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('images/05_02.png', dpi=300)
plt.show()

# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w)

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=3)

X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

print(X_train_pca.shape)
print(X_test_pca.shape)

import mpl_toolkits.mplot3d as p3d
fig=plt.figure()
ax=p3d.Axes3D(fig)
ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1],X_train_pca[:, 2],s=10,alpha=0.2)
ax.set_zlabel('PC 3',fontdict={'color': 'red'})
ax.set_ylabel('PC 2',fontdict={'color': 'red'})
ax.set_xlabel('PC 1',fontdict={'color': 'red'})
#eigenvalues
cov_mat = np.cov(X_train_pca.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
#variance
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

m=np.linspace(1,3,3)
import matplotlib.pyplot as plt
plt.bar(m, var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(m, cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('images/05_02.png', dpi=300)
plt.show()

#linear regression
#orignial
from sklearn.linear_model import LinearRegression
reg_all = LinearRegression()
reg_all.fit(X_train,y_train)
y_train_pred = reg_all.predict(X_train)
y_test_pred = reg_all.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train, c='pink',marker='.',label='train')
plt.scatter(y_test_pred, y_test_pred - y_test, c='green',marker='.',label='test')
plt.xlabel('Predicted values')
plt.ylabel('Residuals Error')
plt.title('Linear Regression(original data)')
plt.legend(loc='upper left')
plt.hlines(0, 0, 13,color="black")
plt.figure()
plt.show()
#MSE,R2
from sklearn.metrics import mean_squared_error
mse_linear=mean_squared_error(y_test,y_test_pred)
r2_linear=reg_all.score(X_test,y_test)
print("MSE of testing data in linear regression is: ",mse_linear)
print("R2 of testing data in linear regression is: ",r2_linear)

mse_linear=mean_squared_error(y_train,y_train_pred)
r2_linear=reg_all.score(X_train,y_train)
print("MSE of trainging data in linear regression is: ",mse_linear)
print("R2 of training data in linear regression is: ",r2_linear)

#pca dataset
reg= LinearRegression()
reg.fit(X_train_pca,y_train)
y_train_pred2 = reg.predict(X_train_pca)
y_test_pred2 = reg.predict(X_test_pca)
plt.scatter(y_train_pred2, y_train_pred2 - y_train, c='pink',marker='.',label='train')
plt.scatter(y_test_pred2, y_test_pred2 - y_test, c='green',marker='.',label='test')
plt.xlabel('Predicted values')
plt.ylabel('Residuals Error')
plt.title('Linear Regression(transformed data)')
plt.legend(loc='upper left')
plt.hlines(0, 0, 13,color="black")
plt.figure()
plt.show()
#MSE,R2
from sklearn.metrics import mean_squared_error
mse_linear2=mean_squared_error(y_test,y_test_pred2)
r2_linear2=reg.score(X_test_pca,y_test)
print("MSE of testing data in linear regression is: ",mse_linear2)
print("R2 of testing data in linear regression is: ",r2_linear2)

mse_linear2=mean_squared_error(y_train,y_train_pred2)
r2_linear2=reg.score(X_train_pca,y_train)
print("MSE of trainging data in linear regression is: ",mse_linear2)
print("R2 of training data in linear regression is: ",r2_linear2)

#svm
#original 
from sklearn.svm import SVR
linear_svr=SVR(kernel='linear')
linear_svr.fit(X_train,y_train)
y_train_pred = linear_svr.predict(X_train)
y_test_pred = linear_svr.predict(X_test)
y_train_pred=y_train_pred.reshape(y_train_pred.shape[0],1)
y_test_pred=y_test_pred.reshape(y_test_pred.shape[0],1)
plt.scatter(y_train_pred, y_train_pred - y_train, c='pink',marker='.',label='train')
plt.scatter(y_test_pred, y_test_pred - y_test, c='green',marker='.',label='test')
plt.xlabel('Predicted values')
plt.ylabel('Residuals Error')
plt.title('SVM Regression(original data)')
plt.legend(loc='upper left')
plt.hlines(0, 0, 13,color="black")
plt.figure()
plt.show()
#MSE,R2
from sklearn.metrics import mean_squared_error
mse_linear=mean_squared_error(y_test,y_test_pred)
r2_linear=linear_svr.score(X_test,y_test)
print("MSE of testing data in SVM regression is: ",mse_linear)
print("R2 of testing data in SVM regression is: ",r2_linear)

mse_linear=mean_squared_error(y_train,y_train_pred)
r2_linear=linear_svr.score(X_train,y_train)
print("MSE of trainging data in SVM regression is: ",mse_linear)
print("R2 of training data in SVM regression is: ",r2_linear)

#pca
linear_svr=SVR(kernel='linear')
linear_svr.fit(X_train_pca,y_train)
y_train_pred = linear_svr.predict(X_train_pca)
y_test_pred = linear_svr.predict(X_test_pca)
y_train_pred=y_train_pred.reshape(y_train_pred.shape[0],1)
y_test_pred=y_test_pred.reshape(y_test_pred.shape[0],1)
plt.scatter(y_train_pred, y_train_pred - y_train, c='pink',marker='.',label='train')
plt.scatter(y_test_pred, y_test_pred - y_test, c='green',marker='.',label='test')
plt.xlabel('Predicted values')
plt.ylabel('Residuals Error')
plt.title('SVM Regression(transformed data)')
plt.legend(loc='upper left')
plt.hlines(0, 0, 13,color="black")
plt.figure()
plt.show()
#MSE,R2
from sklearn.metrics import mean_squared_error
mse_linear=mean_squared_error(y_test,y_test_pred)
r2_linear=linear_svr.score(X_test_pca,y_test)
print("MSE of testing data in SVM regression is: ",mse_linear)
print("R2 of testing data in SVM regression is: ",r2_linear)

mse_linear=mean_squared_error(y_train,y_train_pred)
r2_linear=linear_svr.score(X_train_pca,y_train)
print("MSE of trainging data in SVM regression is: ",mse_linear)
print("R2 of training data in SVM regression is: ",r2_linear)



print("My name is {Yingshan He}")
print("My NetID is: {yh29}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


