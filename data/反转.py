#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 09:33:13 2023

@author: kai
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import pickle
close=pd.read_excel('调仓收盘价.xlsx',skiprows=1,index_col=0)
#%%
with open('周期收益率.pkl', 'rb') as file:
    period_profit = pickle.load(file)
#%%反转
with open('dailyreturn.pkl', 'rb') as file:
    dailyreturn = pickle.load(file)

zhenfu=pd.read_excel('反转.xlsx',skiprows=1,index_col=0)
#%%
mmt2=[]
groupsize=4
for i in range(len(zhenfu)):
    if zhenfu.index[i] in close.index.values:
        mmt=[]
        for k in range(0,1417):           
            s=zhenfu.iloc[0:i,k]
            e=s[s.ne(0)].tail(20)
            if len(e)==0:
                mmt.append(0)
            else:
                sort=e.sort_values(ascending=False)
                group=[sort.iloc[j:j+4] for j in range(0, len(sort), 4)]
                list1=group[0].index
                list2=group[-1].index
                mean1=np.mean(dailyreturn.loc[list1].iloc[:,k])
                mean2=np.mean(dailyreturn.loc[list2].iloc[:,k])
                mean=mean1-mean2
                mmt.append(mean)
        mmt2.append(mmt)
    
reverse=pd.DataFrame(mmt2,columns=zhenfu.columns)
#mmt2.replace([0, 0],0,inplace=True)

#%%
ic2=reverse.corrwith(period_profit,axis=0)
#plt.scatter(profit2.iloc[:,1],turnstd.iloc[:,1])
ic=np.mean(ic2)
icir=np.mean(ic2)/np.std(ic2)
print(ic,icir)
#%%量价
turn=pd.read_excel('换手率.xlsx',skiprows=1,index_col=0)
dailyprice=pd.read_excel('日收盘价.xlsx',skiprows=1,index_col=0)
#%%
info=[]
for i in range(len(turn)):
    if turn.index[i] in close.index.values:
        inf=[]
        for k in range(0,1417):  
            p1=pd.DataFrame(turn.iloc[i-20:i,k])
            p2=dailyprice.shift(-1).iloc[i-20+20:i+20,k]
            #p1['2']=p2
            pearson=p1.corrwith(p2)
            inf.append(pearson[0])
        info.append(inf)
    
corre=pd.DataFrame(info,columns=zhenfu.columns)

corre.iloc[0,:]=re
corre.fillna(0,inplace=True)
#%%
with open('反转因子.pkl', 'wb') as file:
    pickle.dump(fanzhuan, file)
#%%
ic2=period_profit.corrwith(corre.iloc[0:144],axis=0)
#plt.scatter(profit2.iloc[:,1],turnstd.iloc[:,1])
ic=np.mean(ic2)
icir=np.mean(ic2)/np.std(ic2)
print(ic,icir)    
#%%
with open('反转.pkl', 'rb') as file:
    reverse = pickle.load(file)
with open('量价.pkl', 'rb') as file:
    corre = pickle.load(file)
#%%

def stan(x):
    mean=x.mean()
    std=x.std()
    return (x-mean)/std

for i in range(145):
    reverse.iloc[i,:]=stan(reverse.iloc[i,:])
    
for i in range(145):
    corre.iloc[i,:]=stan(corre.iloc[i,:])

#%%合成反转
fanzhuan=-reverse-corre#+liudong

ic2=fanzhuan.corrwith(period_profit,axis=0)
#plt.scatter(profit2.iloc[:,1],turnstd.iloc[:,1])
ic=np.mean(ic2)
icir=np.mean(ic2)/np.std(ic2)
print(ic,icir)
#%%低
cang=[]
for i in range(145):
    k = np.min([i//6,23])
    fanzhuan2=fanzhuan.loc[:,stock[k]]
    ld=fanzhuan2.iloc[i,:].sort_values(ascending=False)
    lis=ld.index[0:50]
    cang.append(lis.to_list())


money=1000
debt=2
low2=[]
close.index=np.arange(145)
for i in range(0,144):
    data=close.loc[i:i+1,cang[i]]
    ret=data.iloc[-1,:]/data.iloc[0,:]
    mon=ret*money/len(ret)
    money=np.sum(mon)
    
    equal=pd.Series([1]*len(ret),index=cang[i+1])
    target=equal*money/len(ret)
    exp=abs(target.sub(mon,fill_value=0))
    exp=np.sum(exp)*(2/1000)
    debt+=exp
    low2.append(money)
    print(money,debt)
#%%正常

cang=[]
for i in range(145):
    ld=fanzhuan.iloc[i,:].sort_values(ascending=False)
    lis=ld.index[0:10]
    cang.append(lis.to_list())

money=1000
debt=2
normal2=[]
close.index=np.arange(145)
for i in range(0,144):
    data=close.loc[i:i+1,cang[i]]
    ret=data.iloc[-1,:]/data.iloc[0,:]
    mon=ret*money/len(ret)
    money=np.sum(mon)
    
    equal=pd.Series([1]*len(ret),index=cang[i+1])
    target=equal*money/len(ret)
    exp=abs(target.sub(mon,fill_value=0))
    exp=np.sum(exp)*(2/1000)
    debt+=exp
    normal2.append(money)
    print(money,debt)
#%%
zhongzheng.index=np.arange(145)
normal2=pd.DataFrame(normal2)
low2=pd.DataFrame(low2)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.plot(normal2,c='green',label='全股票')
plt.plot(low2,c='red',label='低关注度')
plt.plot(zhongzheng,label='基准')
plt.legend()
plt.show()
