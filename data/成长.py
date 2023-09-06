#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 10:12:16 2023

@author: kai
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import pickle
#close=pd.read_excel('调仓收盘价.xlsx',skiprows=1,index_col=0)
#%%
size=pd.read_excel('营业利润.xlsx',sheet_name=0, skiprows=1,index_col=0)
acce=size.diff().copy()
size2=pd.read_excel('利润增速.xlsx',sheet_name=0, skiprows=1,index_col=0)
size=size.iloc[3:-1,:]
size2=size2.iloc[0:-2,:]
#%%
'''
size.index=np.arange(144)
size2.index=np.arange(144)
#%%残差
size2['constant']=1
sizes=size.copy()
for i in range(1417):
    model=sm.OLS(sizes.iloc[:,i],size2.iloc[:,[i,-1]])
    result=model.fit()
    sizes.iloc[:,i]=result.resid
    '''
#%%acc
acc=acce.iloc[3:-1,:]
acc=acc.replace(0, pd.NA).fillna(method='ffill')
acc.index=np.arange(144)
size3=size.copy()
size3.index=np.arange(144)

for i in range(144):
    size3.iloc[i,:]=size3.iloc[i,:].rank(ascending=True,pct=True)
    '''
    for k in range(1417):
        if size3.iloc[i,k]<0.33:
            size3.iloc[i,k]=1
        elif size3.iloc[i,k]<0.66 and size3.iloc[i,k]>0.33:
            size3.iloc[i,k]=2
        else:
            size3.iloc[i,k]=3
      '''      
for i in range(144):
    acc.iloc[i,:]=acc.iloc[i,:].rank(ascending=True,pct=True)
    '''
    for k in range(1417):
        if acc.iloc[i,k]<0.33:
            acc.iloc[i,k]=1
        elif acc.iloc[i,k]<0.66 and acc.iloc[i,k]>0.33:
            acc.iloc[i,k]=2
        else:
            acc.iloc[i,k]=3
    '''
QPT=acc+size3
#%%
#%%
def stan(x):
    mean=x.mean()
    std=x.std()
    return (x-mean)/std

for i in range(144):
    size.iloc[i,:]=stan(size.iloc[i,:])

size.fillna(0,inplace=True)
#%%  

ic2=profit.corrwith(QPT,axis=0)

ic=np.mean(ic2)
icir=np.mean(ic2)/np.std(ic2)
print(ic,icir) 
#%%
with open('qpt因子.pkl', 'wb') as file:
    pickle.dump(QPT, file)