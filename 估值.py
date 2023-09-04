#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 14:28:27 2023

@author: kai
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import pickle
#%%
'''
value=pd.read_excel('市盈率.xlsx',skiprows=1,index_col=0)
value.index=date
value.drop('20220701',inplace=True)
value=1/value
#value=-value
#%%
with open('价值因子.pkl', 'wb') as file:
    pickle.dump(value, file)
'''
#%%
pb=pd.read_excel('PBROE.xlsx',sheet_name=0,skiprows=1,index_col=0)
roe=pd.read_excel('PBROE.xlsx',sheet_name=1,skiprows=1,index_col=0)
roe.fillna(method='ffill',inplace=True)
pb.fillna(method='ffill',inplace=True)
bp=1/pb
pb.fillna(0,inplace=True)
roe.fillna(0,inplace=True)
#%%
bp.index=profit2.index
roe.index=profit2.index
pb.index=profit2.index
ic2=profit2.corrwith(bp,axis=0)

ic=np.mean(ic2)
icir=np.mean(ic2)/np.std(ic2)
print(ic,icir) 
#%%
for i in range(144):
    model=sm.OLS(pb.iloc[i,:],sm.add_constant(roe.iloc[i,:])).fit()
    pb.iloc[i,:]=model.resid
pb=-pb
#%%
with open('价值因子.pkl', 'wb') as file:
    pickle.dump(pb, file)