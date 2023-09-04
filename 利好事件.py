#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 15:46:40 2023

@author: kai
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import pickle
import alphalens
import math
#%%股权激励
prices=pd.read_excel('价格.xlsx',sheet_name=0,skiprows=1,index_col=0)
base=pd.read_excel('中信一级.xlsx',skiprows=1,names=['Date','index'],index_col=0)
#%%
enc=pd.read_excel('股权激励一览.xlsx',sheet_name=0,skiprows=0,index_col=1)
enc.drop(['序号','名称','最新公告日期'],axis=1,inplace=True)
enc=enc.drop_duplicates()
enc.dropna(inplace=True)

low=[]
high=[]
for idx,datet in enc.iterrows():
    start=datet.iloc[0]#+pd.offsets.BDay(1)
    pri=prices[idx].loc[start:].iloc[:121]
    albase=base.loc[start:].iloc[:121]
    pri=pri/pri.iloc[0]
    albase=albase/albase.iloc[0]
    excess=pri-albase['index']
    excess.name=pri.name
    for k in range(9,24):
        if datet.iloc[0]>date[6*k] and datet.iloc[0]<date[6*k+6]:
            if idx in stock[k]:
                excess.index=np.arange(len(excess))
                low.append(excess)
            else:
                excess.index=np.arange(len(excess))
                high.append(excess)  
highlist=pd.concat(high,axis=1)
lowlist=pd.concat(low,axis=1)
lowmean=lowlist.mean(axis=1)
highmean=highlist.mean(axis=1)
plt.plot(lowmean,c='red')
print(len(lowlist.T))
plt.plot(highmean)
print(len(highlist.T))
plt.show()

short=(lowlist.iloc[2,:])
big=(abs(short)>0.02)
small=(abs(short)<0.02)
bigpool=lowlist.loc[:,big]
smallpool=lowlist.loc[:,small]
plt.plot(bigpool.T.mean())
plt.plot(smallpool.T.mean(),c='red')
plt.show()

quantiles=short.quantile([0, 0.2, 0.4, 0.6,0.8, 1.0])
newlow = []
for i in range(5):
    lower_bound = quantiles.iloc[i]
    upper_bound = quantiles.iloc[i+1]
    mask = (short >= lower_bound) & (short <= upper_bound)
    filtered=lowlist.loc[:,mask]
    newlow.append(filtered.T.mean())
plt.plot(newlow[0],label='1')
plt.plot(newlow[1],label='2')
plt.plot(newlow[2],label='3')
plt.plot(newlow[3],label='4')
plt.plot(newlow[4],label='5')
jili=smallpool
plt.legend()

plt.show()

#%%员工持股
emp=pd.read_excel('员工持股计划.xlsx',sheet_name=0,index_col=1)
emp.drop(['序号','证券简称','股东大会公告日'],axis=1,inplace=True)
emp=emp.drop_duplicates()
emp.dropna(inplace=True)

low=[]
high=[]
for idx,datet in emp.iterrows():
    start=datet.iloc[0]
    pri=prices[idx].loc[start:].iloc[:121]
    albase=base.loc[start:].iloc[:121]
    pri=pri/pri.iloc[0]
    albase=albase/albase.iloc[0]
    excess=pri-albase['index']
    excess.name=pri.name
    for k in range(9,24):
        if start>date[6*k] and start<date[6*k+6]:
            if idx in stock[k]:
               excess.index=np.arange(len(excess))
               low.append(excess)
            else:
               excess.index=np.arange(len(excess))
               high.append(excess) 
             
highlist=pd.concat(high,axis=1)
lowlist=pd.concat(low,axis=1)
lowmean=lowlist.mean(axis=1)
highmean=highlist.mean(axis=1)
plt.plot(lowmean,c='red')
plt.plot(highmean)
plt.show()

short=lowlist.iloc[2,:]
big=(short >0) 
small=(short<0) #& (short<0.02)
bigpool=lowlist.loc[:,big]
smallpool=lowlist.loc[:,small]
plt.plot(bigpool.T.mean(),c='red')
plt.plot(smallpool.T.mean())
empl=bigpool
plt.show()

#%%增发
inc=pd.read_excel('定向增发.xlsx',sheet_name=0,index_col=1)
inc.drop(['名称','序号','发行日期','定增股份上市日','初始预案公告日'],axis=1,inplace=True)
inc=inc.drop_duplicates()
inc.dropna(inplace=True)

low=[]
high=[]
for idx,datet in inc.iterrows():
    start=datet.iloc[0]
    pri=prices[idx].loc[start:].iloc[:121]
    albase=base.loc[start:].iloc[:121]
    pri=pri/pri.iloc[0]
    albase=albase/albase.iloc[0]
    excess=pri-albase['index']
    excess.name=pri.name
    for k in range(9,24):
        if start>date[6*k] and start<date[6*k+6]:
            if idx in stock[k]:
               excess.index=np.arange(len(excess))
               low.append(excess)
            else:
               excess.index=np.arange(len(excess))
               high.append(excess) 
             
highlist=pd.concat(high,axis=1)
#highlist-=1
lowlist=pd.concat(low,axis=1)
#lowlist-=1
lowmean=lowlist.mean(axis=1)
highmean=highlist.mean(axis=1)
plt.plot(lowmean,c='red')
print(len(lowlist.T))
plt.plot(highmean)
print(len(highlist.T))
plt.show()

short=lowlist.iloc[2,:]
big=(short >0) 
small=(short<0) #& (short<0.02)
bigpool=lowlist.loc[:,big]
smallpool=lowlist.loc[:,small]
plt.plot(bigpool.T.mean(),c='red')
plt.plot(smallpool.T.mean())
'''
quantiles=short.quantile([0, 0.25, 0.5, 0.75, 1.0])
newlow = []
for i in range(4):
    lower_bound = quantiles.iloc[i]
    upper_bound = quantiles.iloc[i+1]
    mask = (short >= lower_bound) & (short <= upper_bound)
    filtered=lowlist.loc[:,mask]
    newlow.append(filtered.T.mean())
plt.plot(newlow[0],label='1')
plt.plot(newlow[1],label='2')
plt.plot(newlow[2],label='3')
plt.plot(newlow[3],label='4')
'''
zengfa=bigpool
plt.show()


#%%
plt.plot(jili.T.mean())
plt.plot(zengfa.T.mean())
plt.plot(empl.T.mean())
#%%
testdate=pd.date_range(start='20141230',end='20220630',freq='BM')
testdate=testdate.to_list()
testdate[25]=testdate[25]+datetime.timedelta(-5)
testdate[40]=testdate[40]+datetime.timedelta(-3)      
testdate[48]=testdate[48]+datetime.timedelta(-3)
testdate[61]=testdate[61]+datetime.timedelta(-8)
testdate[85]=testdate[85]+datetime.timedelta(-3)
#print(testdate)
off=datetime.timedelta(180)
zengfariqi=inc.loc[zengfa.columns]
jiliriqi=enc.loc[jili.columns]
empriqi=emp.loc[empl.columns]
jiliriqi.columns=['公告日']

cang=[[] for i in range(84)]
for i in range(84):
    
    for j in range(len(zengfariqi)):
        q=zengfariqi.iloc[j]
        if q[0] < testdate[6+i] and q[0] >= testdate[i+6]-off:
            cang[i].append(q.name)
            
    for k in range(len(jiliriqi)):
        w=jiliriqi.iloc[k]
        if w[0] < testdate[6+i] and w[0] >= testdate[i+6]-off:
            cang[i].append(w.name)

    for z in range(len(empriqi)):
        o=empriqi.iloc[z]
        if o[0] < testdate[6+i] and o[0] >= testdate[i+6]-off:
            cang[i].append(o.name)


priceevent=prices.loc[testdate[6:]]
money=1000
debt=0
low=[1000]
#closep.index=np.arange(145)
for i in range(0,84):
    data=priceevent.loc[testdate[i+6]:testdate[i+7],cang[i]]
    ret=data.iloc[-1,:]/data.iloc[0,:]
    mon=ret*money/len(cang[i])
    money=np.sum(mon)
    if i<83:
        equal=pd.Series([1]*len(cang[i+1]),index=cang[i+1])
        target=equal*money/len(cang[i+1])
        exp=abs(target.sub(mon,fill_value=0))
        exp=np.sum(exp)*(2/1000)/2 
    if i==83:
        exp=money*2/1000
    debt+=exp
    money-=exp
    low.append(money)
    print(money,debt)

pr=(money/1000)**(1/7)-(base.loc[testdate[90]][0]/base.loc[testdate[6]][0])**(1/7)
print('年化超额收益率：',100*pr)

plt.rcParams['font.sans-serif'] = ['SimHei']
low=pd.DataFrame(low,index=testdate[6:])
plt.plot(low,c='red',label='利好事件组合')
baseplot=1000*base.loc[testdate[6]:]/base.loc[testdate[6]][0]
plt.plot(baseplot,label='中信一级指数')

plt.legend()
plt.show()
#%%组合
newcombi=[]
lowt=[]
normalt=[]

ss3=ss.iloc[60:]
for i in range(84):
    tar=ss3.iloc[i].loc[cang[i]]
    tar.drop_duplicates(inplace=True)
    rank=tar.sort_values(ascending=False)
    candidate=rank.index[0:10]
    newcombi.append(candidate.to_list())
    
    rank2=ss3.iloc[i].sort_values(ascending=False)
    candidate2=rank2.index[0:10]
    normalt.append(candidate2.to_list())
    
    rank3=ss3.iloc[i].loc[stock[i//6+10]].sort_values(ascending=False)
    candidate3=rank3.index[0:10]
    lowt.append(candidate3.to_list())

priceevent=prices.loc[testdate[6:]]
#%%
money=1000
debt=0
new=[1000]
#closep.index=np.arange(145)
for i in range(0,84):
    data=priceevent.loc[testdate[i+6]:testdate[i+7],newcombi[i]]
    ret=data.iloc[-1,:]/data.iloc[0,:]
    mon=ret*money/len(newcombi[i])
    money=np.sum(mon)
    if i<83:
        equal=pd.Series([1]*len(newcombi[i+1]),index=newcombi[i+1])
        target=equal*money/len(newcombi[i+1])
        exp=abs(target.sub(mon,fill_value=0))
        exp=np.sum(exp)*(2/1000)/2 
    if i==83:
        exp=money*2/1000
    debt+=exp
    money-=exp
    new.append(money)
    #print(money,debt)

pr=(money/1000)**(1/7)-(base.loc[testdate[90]][0]/base.loc[testdate[6]][0])**(1/7)
print('年化超额收益率：',100*pr)

new=pd.DataFrame(new,index=testdate[6:])
plt.plot(new,c='green',label='利好组合叠加因子')

money=1000
debt=0
low=[1000]
#closep.index=np.arange(145)
for i in range(0,84):
    data=priceevent.loc[testdate[i+6]:testdate[i+7],lowt[i]]
    ret=data.iloc[-1,:]/data.iloc[0,:]
    mon=ret*money/len(lowt[i])
    money=np.sum(mon)
    if i<83:
        equal=pd.Series([1]*len(lowt[i+1]),index=lowt[i+1])
        target=equal*money/len(lowt[i+1])
        exp=abs(target.sub(mon,fill_value=0))
        exp=np.sum(exp)*(2/1000)/2 
    if i==83:
        exp=money*2/1000
    debt+=exp
    money-=exp
    low.append(money)
    #print(money,debt)

pr=(money/1000)**(1/7)-(base.loc[testdate[90]][0]/base.loc[testdate[6]][0])**(1/7)
print('年化超额收益率：',100*pr)

low=pd.DataFrame(low,index=testdate[6:])
plt.plot(low,c='red',label='因子模型（低关注度)')

money=1000
debt=0
normal=[1000]
#closep.index=np.arange(145)
for i in range(0,84):
    data=priceevent.loc[testdate[i+6]:testdate[i+7],normalt[i]]
    ret=data.iloc[-1,:]/data.iloc[0,:]
    mon=ret*money/len(normalt[i])
    money=np.sum(mon)
    if i<83:
        equal=pd.Series([1]*len(normalt[i+1]),index=normalt[i+1])
        target=equal*money/len(normalt[i+1])
        exp=abs(target.sub(mon,fill_value=0))
        exp=np.sum(exp)*(2/1000)/2 
    if i==83:
        exp=money*2/1000
    debt+=exp
    money-=exp
    normal.append(money)
    #print(money,debt)

pr=(money/1000)**(1/7)-(base.loc[testdate[90]][0]/base.loc[testdate[6]][0])**(1/7)
print('年化超额收益率：',100*pr)

normal=pd.DataFrame(normal,index=testdate[6:])
#plt.plot(normal,label='因子模型（全股票池）')

baseplot=1000*base.loc[testdate[6]:testdate[-1]]/base.loc[testdate[6]][0]
plt.plot(baseplot,label='中信一级指数')
plt.legend()
plt.show()
#%%叠加策略
net=new
marvo=baseplot.loc[net.index].pct_change()
marvo.drop(['20150630'],inplace=True)
#marvo.index=date2

returns=net.pct_change()
returns.drop(['20150630'],inplace=True)

cova=np.cov(returns.T,marvo.T)
beta=cova[0,1]/cova[1,1]
print('beta:',beta)

ann=((net.iloc[-1][0]/1000)**(1/7)-1)*100
baseann=((baseplot.iloc[-1][0]/1000)**(1/7)-1)*100
alpha=ann-4-beta*(baseann-4)
print('alpha:',alpha)

sharpe=(ann/100-0.04)/(np.sqrt(7)*np.std(returns.iloc[:,0]))
print('sharpe ratio:',sharpe)

