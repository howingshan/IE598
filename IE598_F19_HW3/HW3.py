#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 12:01:27 2019

@author: howingshan
"""


import sys
import numpy as np
with open('/Users/howingshan/Desktop/2019FALL/IE598/HW3/HY.csv','r') as f:
    print(f.read())
    
xList = []
labels = []
for line in data:
    #split on comma
    row = line.strip().split(",")
    xList.append(row)
nrow = len(xList)
ncol = len(xList[1])

type = [0]*3
colCounts = []
for col in range(ncol):
    for row in xList:
        try:
            a = float(row[col])
            if isinstance(a, float):
                type[0] += 1
        except ValueError:
            if len(row[col]) > 0:
                type[1] += 1
            else:
                type[2] += 1
    colCounts.append(type)
    type = [0]*3
    
sys.stdout.write("Col#" + '\t' + "Number" + '\t' +
                 "Strings" + '\t ' + "Other\n")
iCol = 0
for types in colCounts:
    sys.stdout.write(str(iCol) + '\t\t' + str(types[0]) + '\t\t' +
                     str(types[1]) + '\t\t' + str(types[2]) + "\n")
iCol += 1

colData = []
for row in xList:
    colData.append(float(row[col]))
colArray = np.array(colData)
colMean = np.mean(colArray)
colsd = np.std(colArray)
sys.stdout.write("Mean = " + '\t' + str(colMean) + '\t\t' +
            "Standard Deviation = " + '\t ' + str(colsd) + "\n")
#calculate quantile boundaries
ntiles = 4
percentBdry = []
for i in range(ntiles+1):
    percentBdry.append(np.percentile(colArray, i*(100)/ntiles))
sys.stdout.write("\nBoundaries for 4 Equal Percentiles \n")
print(percentBdry)
sys.stdout.write(" \n")

#run again with 10 equal intervals
ntiles = 10
percentBdry = []
for i in range(ntiles+1):
    percentBdry.append(np.percentile(colArray, i*(100)/ntiles))
sys.stdout.write("Boundaries for 10 Equal Percentiles \n")
print(percentBdry)
sys.stdout.write(" \n")
#The last column contains categorical variables
col = 29
colData = []
for row in xList:
    colData.append(row[col])
unique = set(colData)
sys.stdout.write("Unique Label Values \n")
print(unique)

#count up the number of elements having each value
catDict = dict(zip(list(unique),range(len(unique))))
catCount = [0]*5
for elt in colData:
    catCount[catDict[elt]] += 1

colData = []
for row in xList:
   colData.append(float(row[col]))
import pylab
import scipy.stats as stats
stats.probplot(colData, dist="norm", plot=pylab)
pylab.show()

import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plot

df=pd.read_csv('/Users/howingshan/Desktop/2019FALL/IE598/HW3/HY.csv',header=None)
print(df.head())
print(df.tail())
summary=df.describe()
print(summary)
df=df[['bond_type']].astype('int')

for i in range(2722):
    #assign color based on "M" or "R" labels
    if df.iat[i,29] == '1':
        pcolor = "red"
    elif df.iat[i,29] == '2':
        pcolor = "purple" 
    elif df.iat[i,29] == '3':
        pcolor = "pink"
    elif df.iat[i,29] == '4':
        pcolor = "green" 
    else:
        pcolor = "blue"
          #plot rows of data as if they were series data
    dataRow = df.iloc[i,0:36]
    dataRow.plot(color=pcolor)
plot.xlabel("bond_type")
plot.ylabel(("number"))
plot.show()

dataRow25 = df.iloc[:,25]
dataRow26 = df.iloc[:,26]
plot.scatter(dataRow26, dataRow25)
plot.xlabel("days_diff_max")
plot.ylabel(("n_days_trade"))
plot.show()

dataRow31 = df.iloc[:,31]
plot.scatter(dataRow26, dataRow31)
plot.xlabel("days_diff_max")
plot.ylabel(("weekly_mean_volume"))
plot.show()


target=[]
for i in range(2272):
    if df.iat[i,29] == '1':
        target.append(1.0) 
    elif df.iat[i,29] == '2':
        target.append(2.0) 
    elif df.iat[i,29] == '3':
        target.append(3.0) 
    elif df.iat[i,29] == '4':
        target.append(4.0) 
    else:
        target.append(5.0)
        
dataRow = df.iloc[0:2272,17]
plot.scatter(dataRow, target)
plot.xlabel("Months in HYG")
plot.ylabel("bond_type")
plot.show()

from random import uniform
target = []
for i in range(2272):
#assign 0 or 1 target value based on "M" or "R" labels
    # and add some dither
    if df.iat[i,29] == '1':
        target.append(1.0 + uniform(-0.1, 0.1))
    elif df.iat[i,29] == '2':
        target.append(2.0 + uniform(-0.1, 0.1))
    elif df.iat[i,29] == '3':
        target.append(3.0 + uniform(-0.1, 0.1))
    elif df.iat[i,29] == '4':
        target.append(4.0 + uniform(-0.1, 0.1))
        
    else:
        target.append(5.0 + uniform(-0.1, 0.1))
    #plot 35th attribute with semi-opaque points
dataRow = df.iloc[0:2272,17]
plot.scatter(dataRow, target, alpha=0.5, s=120)
plot.xlabel("Months in HYG")
plot.ylabel("bond_type")
plot.show()

from math import sqrt
from pandas import DataFrame
dataRow25 = df.iloc[1:,25]
dataRow26 = df.iloc[1:,26]
dataRow31 = df.iloc[1:,31]
dataRow25=dataRow25.astype(float)
dataRow26=dataRow26.astype(float)
dataRow31=dataRow31.astype(float)
mean25 = 0.0; mean26 = 0.0; mean31 = 0.0
numElt = len(dataRow25)
for i in range(1,2271):
    mean25 += dataRow25[i]/2271
    mean26 += dataRow26[i]/2271
    mean31 += dataRow31[i]/2271
var25 = 0.0; var26 = 0.0; var31 = 0.0
for i in range(1,2271):
    var25 += (dataRow25[i] - mean25) * (dataRow25[i] - mean25)/2271
    var31 += (dataRow31[i] - mean31) * (dataRow31[i] - mean31)/2271
    var26 += (dataRow26[i] - mean26) * (dataRow26[i] - mean26)/2271
corr2625 = 0.0; corr2631 = 0.0
for i in range(1,2271):
    corr2625 += (dataRow26[i] - mean26) * \
              (dataRow25[i] - mean25) / (sqrt(var26*var25) * 2271)
    corr2631 += (dataRow26[i] - mean26) * \
               (dataRow31[i] - mean31) / (sqrt(var26*var31) * 2271)
sys.stdout.write("Correlation between days_diff_max and n_days_trade \n")
print(corr2625)
sys.stdout.write(" \n")
sys.stdout.write("Correlation between days_diff_max and weekly_mean_volume \n")
print(corr2631)
sys.stdout.write(" \n")

corMat = DataFrame(df.corr())
plot.pcolor(corMat)
plot.show()




print("My name is {Yingshan He}")
print("My NetID is: {yh29}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

