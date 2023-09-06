#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 09:15:44 2023

@author: kai
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import pickle
#close=pd.read_excel('调仓收盘价.xlsx',skiprows=1,index_col=0)
#%%
size=pd.read_excel('规模.xlsx',skiprows=1,index_col=0)
#%%
def stan(x):
    mean=x.mean()
    std=x.std()
    return (x-mean)/std

for i in range(144):
    size.iloc[i,:]=stan(size.iloc[i,:])
#%%
size.fillna(0,inplace=True)
#%%  
size.index=profit.index
ic2=profit.corrwith(size,axis=0)

ic=np.mean(ic2)
icir=np.mean(ic2)/np.std(ic2)
print(ic,icir) 
#%%
with open('规模因子.pkl', 'wb') as file:
    pickle.dump(size, file)