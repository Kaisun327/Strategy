#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:03:48 2023

@author: kai
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import pickle
from scipy.optimize import minimize, LinearConstraint,\
    NonlinearConstraint, differential_evolution, Bounds
#%%
'''
for i in range(24):
    stock[i] = [item for item in stock[i] if item in close.columns]

with open('持仓.pkl', 'wb') as file:
    pickle.dump(stock, file)
'''
with open('持仓.pkl', 'rb') as file:
    stock = pickle.load(file)

close=pd.read_excel('调仓收盘价.xlsx',skiprows=1)
'''
date=['2010-07-05', '2010-12-31', '2011-06-30', '2011-12-30',
                   '2012-06-29', '2012-12-31', '2013-06-28', '2013-12-31',
                   '2014-06-30', '2014-12-31', '2015-06-30', '2015-12-31',
                   '2016-06-30', '2016-12-30', '2017-06-30', '2017-12-29',
                   '2018-06-29', '2018-12-28', '2019-06-28', '2019-12-31',
                   '2020-06-30', '2020-12-31', '2021-06-30', '2021-12-31']
date=pd.to_datetime(date)'''
#%%
money=1000
for i in range(0,24):
    data=close.loc[6*i:6*i+6,stock[i]]
    ret=data.iloc[-1,:]/data.iloc[0,:]
    mon=ret*money/len(ret)
    money=np.sum(mon)
    print(money)
#%%换手率
turn=pd.read_excel('换手率.xlsx',skiprows=1)
#%%
turnstd=[]
for i in range(len(turn)):
    if turn['Date'][i] in close.index.values:
        list=[]
        a=np.min([20,i])
        for k in range(1,1418):            
            sta=np.std(turn.iloc[i-a:i,k])
            list.append(sta)
        turnstd.append(list)
        
turnstd=pd.DataFrame(turnstd,columns=turn.columns[1:])
#%%

with open('turnstd.pkl', 'wb') as file:
    pickle.dump(turnstd, file)

with open('turnstd.pkl', 'rb') as file:
    turnstd = pickle.load(file)