#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 14:31:13 2023

@author: kai
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import pickle
from scipy.stats import median_abs_deviation as mad
#%%
openp=pd.read_excel('开盘价.xlsx',skiprows=1,index_col=0)
openp.index=np.arange(145)
profit2=openp.pct_change()*100
profit2.drop([0],axis=0,inplace=True)
profit2.index=np.arange(144)
def stan(x):
    mean=x.mean()
    std=x.std()
    return (x-mean)/std

#%%
with open('反转因子.pkl', 'rb') as file:
    reverse = pickle.load(file)
reverse=reverse.iloc[0:-1,:]
with open('规模因子.pkl', 'rb') as file:
    size = pickle.load(file)

with open('流动因子.pkl', 'rb') as file:
    liquidity = pickle.load(file)
liquidity=liquidity.iloc[0:-1,:]
with open('qpt因子.pkl', 'rb') as file:
    qpt = pickle.load(file)
    
with open('价值因子.pkl', 'rb') as file:
    value = pickle.load(file)
  
with open('持仓.pkl', 'rb') as file:
    stock = pickle.load(file)

industry=pd.read_excel('行业.xlsx')
industry=pd.DataFrame(industry.loc[1].drop(['代码'],axis=0))
industry.drop('Code',axis=0,inplace=True)
#industry.set_index('stocks')
#%%
size=1/size
liquidity.fillna(0,inplace=True)
size.fillna(0,inplace=True)
qpt.fillna(0,inplace=True)
reverse.fillna(0,inplace=True)
value.fillna(0,inplace=True)
value.index=np.arange(144)

#去极值
def madf(df,n):
    median=np.median(df)
    sd=mad(df)
    up=median+n*sd
    down=median-n*sd
    return df.clip(down,up)

f=3*1.4826
for i in range(144):
    liquidity.iloc[i,:]=madf(liquidity.iloc[i,:],f)
    size.iloc[i,:]=madf(size.iloc[i,:],f)
    reverse.iloc[i,:]=madf(reverse.iloc[i,:],f)
    qpt.iloc[i,:]=madf(qpt.iloc[i,:],f)
    value.iloc[i,:]=madf(value.iloc[i,:],f)
#市值中性化

for i in range(144):
    r=reverse.iloc[i,:]
    l=liquidity.iloc[i,:]
    q=qpt.iloc[i,:]
    v=value.iloc[i,:]
    s=size.iloc[i,:]
    s.name=None
    r.name=None
    l.name=None
    q.name=None
    v.name=None
    
    e=pd.merge(industry,pd.DataFrame(s),on=industry.index)
    e.set_index('key_0',inplace=True)
    e2=e.groupby(1).transform(stan)
    size.iloc[i,:]=e2.iloc[:,0]
    s=size.iloc[i,:]
    
    a=pd.merge(industry,pd.DataFrame(r),on=industry.index)
    a.set_index('key_0',inplace=True)
    a2=a.groupby(1).transform(stan)
    r=a2.iloc[:,0]
    model1=sm.OLS(r,sm.add_constant(s)).fit()
    r=model1.resid
    reverse.iloc[i,:]=r
    
    b=pd.merge(industry,pd.DataFrame(l),on=industry.index)
    b.set_index('key_0',inplace=True)
    b2=b.groupby(1).transform(stan)
    l=b2.iloc[:,0]
    model2=sm.OLS(l,sm.add_constant(s)).fit()
    l=model2.resid
    liquidity.iloc[i,:]=l
    
    c=pd.merge(industry,pd.DataFrame(q),on=industry.index)
    c.set_index('key_0',inplace=True)
    c2=c.groupby(1).transform(stan)
    q=c2.iloc[:,0]
    model3=sm.OLS(q,sm.add_constant(s)).fit()
    q=model3.resid
    qpt.iloc[i,:]=q
    
    d=pd.merge(industry,pd.DataFrame(v),on=industry.index)
    d.set_index('key_0',inplace=True)
    d2=d.groupby(1).transform(stan)
    v=d2.iloc[:,0]
    model4=sm.OLS(v,sm.add_constant(s)).fit()
    v=model4.resid
    value.iloc[i,:]=v
   
    

   




#%%转换成排名(暂不需要)
'''
a=True
b=False
for i in range(144):
    size.iloc[i,:]=size.iloc[i,:].rank(ascending=a,pct=False)
for i in range(144):
    reverse.iloc[i,:]=reverse.iloc[i,:].rank(ascending=a,pct=False)
for i in range(144):
    liquidity.iloc[i,:]=liquidity.iloc[i,:].rank(ascending=a,pct=False)
for i in range(144):
    qpt.iloc[i,:]=qpt.iloc[i,:].rank(ascending=a,pct=False)
    
#%%
listp=[]
for i in range(1417):
    correlation=pd.concat([reverse.iloc[:,i],size.iloc[:,i],liquidity.iloc[:,i],\
     qpt.iloc[:,i]],axis=1,keys=['reverse','size','liquidity','qpt'])
    pearson=correlation.corr()
    listp.append(pearson)
finalpearson=np.mean(listp,axis=0)
print(finalpearson)
'''
#%%
ic1=profit2.corrwith(reverse)
icm1=np.mean(ic1)
icir1=icm1/np.std(ic1)

ic2=profit2.corrwith(size)
icm2=np.mean(ic2)
icir2=icm2/np.std(ic2)

ic3=profit2.corrwith(liquidity)
icm3=np.mean(ic3)
icir3=icm3/np.std(ic3)

ic4=profit2.corrwith(qpt)
icm4=np.mean(ic4)
icir4=icm4/np.std(ic4)

ic5=profit2.corrwith(value)
icm5=np.mean(ic5)
icir5=icm5/np.std(ic5)

print(icm1,icir1)
print(icm2,icir2)
print(icm3,icir3)
print(icm4,icir4)
print(icm5,icir5)
#%%标准化

for i in range(144):
    reverse.iloc[i,:]=stan(reverse.iloc[i,:])
    
for i in range(144):
    liquidity.iloc[i,:]=stan(liquidity.iloc[i,:])

for i in range(144):
    size.iloc[i,:]=stan(size.iloc[i,:])

for i in range(144):
    qpt.iloc[i,:]=stan(qpt.iloc[i,:])
    
for i in range(144):
    value.iloc[i,:]=stan(value.iloc[i,:])

#%%
ss=reverse+qpt+liquidity+value+size
ict=profit2.corrwith(ss)
icmt=np.mean(ict)
icirt=icmt/np.std(ict)
print(icmt,icirt)

#%%低
cang=[]
for i in range(144):
    k = i//6
    ss2=ss.loc[:,stock[k]]
    ld=ss2.iloc[i,:].sort_values(ascending=False)
    lis=ld.index[0:10]
    cang.append(lis.to_list())


money=1000
debt=0
low=[1000]
openp.index=np.arange(145)
for i in range(0,144):
    data=openp.loc[i:i+1,cang[i]]
    ret=data.iloc[-1,:]/data.iloc[0,:]
    mon=ret*money/len(ret)
    money=np.sum(mon)
    if i<143:
        equal=pd.Series([1]*len(ret),index=cang[i+1])
        target=equal*money/len(ret)
        exp=abs(target.sub(mon,fill_value=0))
        exp=np.sum(exp)*(2/1000)/2 
    if i==143:
        exp=money*2/1000
    debt+=exp
    money-=exp
    low.append(money)
    print(money,debt)

pr=(money/1000)**(1/12)-(zhongzheng.iloc[-1][0]/1000)**(1/12)
print('年化超额收益率：',100*pr)
#%%
zhongzheng=pd.read_excel('中证.xlsx',index_col=0)
zhongzheng=1000*zhongzheng/zhongzheng.iloc[0][0]
zhongzheng.index=netval.index
low=pd.DataFrame(low)
low.index=netval.index
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.plot(netval,c='red',label='做多前十名')
plt.plot(longshort,label='多空组合')
plt.plot(zhongzheng,label='基准')
plt.legend()

plt.show()
#%%
start_date = '2010-07-01'
end_date = '2022-07-01'
date = pd.date_range(start=start_date, end=end_date, freq='MS')
#close.index=date
ss.index=date[0:-1]
#%%
with open('最终因子.pkl', 'wb') as file:
    pickle.dump(ss, file)


    
    
    
    
    
    
    
    
