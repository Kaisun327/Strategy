#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 17:12:35 2023

@author: kai
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import pickle
#%%
close=pd.read_excel('调仓收盘价.xlsx',skiprows=1,index_col=0)
'''
profit=100*close.diff()/close.shift(1)
profit2=profit.copy()
profit2=profit2.iloc[1:]
profit2.index=np.arange(144)'''
with open('周期收益率.pkl', 'rb') as file:
    period_profit = pickle.load(file)
#%%换手率
with open('turnstd.pkl', 'rb') as file:
    turnstd = pickle.load(file)

turnstd2=turnstd.iloc[0:144]

ic2=turnstd2.corrwith(period_profit,axis=0)
#plt.scatter(profit2.iloc[:,1],turnstd.iloc[:,1])
ic=np.mean(ic2)
icir=np.mean(ic2)/np.std(ic2)
print(ic,icir)
#%%价格弹性
amt=pd.read_excel('成交额.xlsx',skiprows=1,index_col=0)
dailyprice=pd.read_excel('日收盘价.xlsx',skiprows=1,index_col=0)
amt.drop(['2010-05-04'],inplace=True)
amtnew=amt.replace(0,method='ffill')
dailyreturn=100*dailyprice.diff()/dailyprice.shift(1)
dailyreturn.drop(['2010-05-04'],inplace=True)
ela=dailyreturn/amtnew
ela=ela.fillna(0)
#%%
bodong=dailyreturn.groupby([dailyreturn.index.year,dailyreturn.index.month]).std()
bodong.index=np.arange(146)
bodong.drop([0,145],inplace=True)
bodong.index=np.arange(144)
#%%
with open('波动率.pkl', 'wb') as file:
    pickle.dump(bodong, file)
#%%
elasticity=[]
for i in range(len(ela)):
    if ela.index[i] in close.index.values:
        list=[]
        for k in range(0,1417):            
            s=np.mean(ela.iloc[i-20:i,k])
            list.append(s)
        elasticity.append(list)
        
elasticity=pd.DataFrame(elasticity,columns=ela.columns)
elasticity=elasticity.abs()
elasticity=elasticity.replace([np.inf], 0)
#amihud=dailyreturn/amt
#%%
with open('elasticity.pkl', 'wb') as file:
    pickle.dump(elasticity, file)
#%%

ic2=elasticity.corrwith(period_profit,axis=0)
#plt.scatter(profit2.iloc[:,1],turnstd.iloc[:,1])
ic=np.mean(ic2)
icir=np.mean(ic2)/np.std(ic2)
print(ic,icir)
#%%波动
low=pd.read_excel('最高最低价.xlsx',skiprows=1,index_col=0,sheet_name='低')
high=pd.read_excel('最高最低价.xlsx',skiprows=1,index_col=0,sheet_name='高')
high=high.iloc[:,0:1417]
highlow=high/low
#highlow=highlow-1
#%%
'''
hlratio=[]
for i in range(len(highlow)):
    if highlow.index[i] in date2:
        list=[]
        for k in range(0,1417):            
            s=np.std(highlow.iloc[i-20:i,k])
            list.append(s)
        hlratio.append(list)
        

hlratio=pd.DataFrame(hlratio,columns=highlow.columns)'''
hlratio=highlow.groupby([highlow.index.year,highlow.index.month]).mean()
hlratio.index=np.arange(146)
hlratio.drop([0,145],inplace=True)
hlratio.index=np.arange(144)
#%%
hlratio.index=np.arange(144)
ic2=hlratio.corrwith(profit2,axis=0)
#plt.scatter(profit2.iloc[:,1],turnstd.iloc[:,1])
ic=np.mean(ic2)
icir=np.mean(ic2)/np.std(ic2)
print(ic,icir)
#%%
with open('turnstd.pkl', 'rb') as file:
    turnstd = pickle.load(file)
with open('hlratio.pkl', 'rb') as file:
    hlratio = pickle.load(file)
with open('elasticity.pkl', 'rb') as file:
    elasticity = pickle.load(file)
#%%合成流动性
def stan(x):
    mean=x.mean()
    std=x.std()
    return (x-mean)/std

for i in range(145):
    elasticity.iloc[i,:]=stan(elasticity.iloc[i,:])
    
for i in range(145):
    turnstd.iloc[i,:]=stan(turnstd.iloc[i,:])
    
for i in range(145):
    hlratio.iloc[i,:]=stan(hlratio.iloc[i,:])
#%%
liudong=elasticity-turnstd-hlratio
#liudong=-turnstd
#liudong.index=close.index
#liudong=liudong.iloc[]
ic2=liudong.corrwith(period_profit,axis=0)
#plt.scatter(profit2.iloc[:,1],turnstd.iloc[:,1])
ic=np.mean(ic2)
icir=np.mean(ic2)/np.std(ic2)
print(ic,icir)
