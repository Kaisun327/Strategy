#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 10:13:09 2023

@author: kai
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import pickle
from scipy.optimize import minimize, LinearConstraint,\
    NonlinearConstraint, differential_evolution, Bounds
#%%growth
'''
ratio=pd.read_excel('data.xlsx',skiprows=3,index_col='Date',\
                    sheet_name='基金持股比例')
    '''
with open('基金持股比.pkl', 'wb') as file:
    pickle.dump(ratio, file)

with open('成交额.pkl', 'wb') as file:
    pickle.dump(amt, file)

with open('持仓.pkl', 'wb') as file:
    pickle.dump(stocks, file)
#%%
amt=pd.read_excel('amt.xlsx',skiprows=1)
amt.fillna(0, inplace=True)
#%%
with open('基金持股比.pkl', 'rb') as file:
    ratio = pickle.load(file)
#%%
date=['2010-07-05', '2010-12-31', '2011-06-30', '2011-12-30',
               '2012-06-29', '2012-12-31', '2013-06-28', '2013-12-31',
               '2014-06-30', '2014-12-31', '2015-06-30', '2015-12-31',
               '2016-06-30', '2016-12-30', '2017-06-30', '2017-12-29',
               '2018-06-29', '2018-12-28', '2019-06-28', '2019-12-31',
               '2020-06-30', '2020-12-31', '2021-06-30', '2021-12-31']
date=pd.to_datetime(date)
#%%
amt2=[]
for i in date:
    for j in amt.index:
        if amt['Date'][j]==i:
            meanlist=[]
            for k in range(1,5243):
                mean=np.mean(amt.iloc[j-120:j,k])
                meanlist.append(mean)
            amt2.append(meanlist)
amt2=pd.DataFrame(amt2,columns=ratio.columns)
#%%
stocks=[]
for i in range(24):
    cang=[]
    for j in range(5242):
        if ratio.iloc[i,j]<1 and amt2.iloc[i][j]>4000:
            cang.append(ratio.columns[j])
    stocks.append(cang)
print(stocks)
#%%
ratio = ratio.reindex(columns=amt.columns)
ratio.drop(columns=['Date'],inplace=True)




