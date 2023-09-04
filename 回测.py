#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 14:40:09 2023

@author: kai
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import pickle
import alphalens
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

#%%
def stan(x):
    mean=x.mean()
    std=x.std()
    return (x-mean)/std

def max_drawdown(returns):
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns / peak) - 1
    return drawdown.min()

openp=pd.read_excel('开盘价.xlsx',skiprows=1,index_col=0)
openp.index=np.arange(145)
profit2=openp.pct_change()*100
profit2.drop([0],axis=0,inplace=True)
profit2.index=np.arange(144)

with open('最终因子.pkl', 'rb') as file:
    ss = pickle.load(file)

with open('持仓.pkl', 'rb') as file:
    stock = pickle.load(file)
#closep=pd.read_excel('调仓收盘价.xlsx',skiprows=1,index_col=0)
start_date = '2010-07-01'
end_date = '2022-07-01'
date = pd.date_range(start=start_date, end=end_date, freq='MS')
openp.index=date
openp.index.set_names(['date'],inplace=True)
#date2 = pd.date_range(start='2010-06-01', end='2022-05-01', freq='MS')
date2 = pd.date_range(start='2010-07-01', end='2022-06-01', freq='MS')


with open('波动率.pkl', 'rb') as file:
    bodong = pickle.load(file)
bodong.index=np.arange(144)
with open('elasticity.pkl', 'rb') as file:
    ela = pickle.load(file)
ela.drop(144,inplace=True)

for i in range(144):
    ela.iloc[i,:]=stan(ela.iloc[i,:])
    
for i in range(144):
    bodong.iloc[i,:]=stan(bodong.iloc[i,:])

risky=ela-bodong
#%%quantile选股
#risky.index=date2
#hlratio.drop(144,inplace=True)
#elasticity.replace(0, np.nan, inplace=True)
openp.index=date
ss.index=date2
months = date2
resultlist=[]
#elasticity.index=ss.index
for idx, month in enumerate(months):
    if idx % 6 ==0:
        stocknum=int(idx/6)
        newss=ss.loc[:,stock[stocknum]]
        '''
        for k in range(6):      
            stable=risky.iloc[idx+k].loc[newss.columns].sort_values(ascending=True)
            deletes=stable.index[len(newss.T)//10:]
        '''
        newss2=newss.iloc[idx:idx+6]
        #newss2=newss2.loc[:,deletes]
        ss1=pd.DataFrame(newss2.stack())
        ss1.index.set_names(['date','asset'],inplace=True)
        ss1.columns=['factor']
        
        newclose=openp.loc[:,stock[stocknum]]
        newclose=newclose.iloc[idx:idx+7]
        
        result=alphalens.utils.get_clean_factor_and_forward_returns(\
                ss1,newclose,quantiles=40,periods=[1],max_loss=1)
        resultlist.append(result)
            
            
result=pd.concat(resultlist)   
quant=result['factor_quantile']
cangtop=[]
cangbot=[]
for i in range(144):
    v=quant.loc[date2[i]]
    d=[]
    e=[]
    for k in range(len(v)):
        if v.iloc[k]==40:
            d.append(v.index[k])
        if v.iloc[k]==1:
            e.append(v.index[k])
    cangtop.append(d)
    cangbot.append(e)
#%%前十名选股
'''
cangtop=[]
cangbot=[]
ranks=10
for i in range(144):
    k = i//6
    ss2=ss.loc[:,stock[k]]
    stable=risky.iloc[i].loc[ss2.columns].sort_values(ascending=True)
    deletes=stable.index[len(ss2.T)//10:]
    ld=ss2.loc[:,deletes].iloc[i,:].sort_values(ascending=False)
    top=ld.index[0:ranks]
    bot=ld.index[-ranks:]
    cangtop.append(top.to_list())
    cangbot.append(bot.to_list())
'''
#%%
alphalens.tears.create_full_tear_sheet(result,long_short=False)

#%%
zhongzheng=pd.read_excel('中证.xlsx',index_col=0,names=['price'])
#zhongzheng=pd.DataFrame(openp.T.sum(),columns=['price'])
zhongzheng.index=date
zhongzheng=1*zhongzheng/zhongzheng.iloc[0]
marvo=zhongzheng.pct_change()
marvo.drop(['20100701'],inplace=True)
#%%多空
from datetime import timedelta
money=1
low=[1]
openp.index=np.arange(145)
shortratio=0.5
longratio=1-shortratio
longm=longratio
shortm=shortratio
long=[longm]
short=[shortm]

for i in range(0,144):
    longda=openp.loc[i:i+1,cangtop[i]]
    shortda=openp.loc[i:i+1,cangbot[i]]
    longret=longda.iloc[-1,:]/longda.iloc[0,:]
    shortret=shortda.iloc[-1,:]/shortda.iloc[0,:]
    shortcap=shortratio*money*shortret/len(shortret)
    longcap=longratio*money*longret/len(longret)
    longpro=(np.sum(longcap)-money*longratio)
    shortpro=money*shortratio-np.sum(shortcap)
    profit=longpro+shortpro
    
    money=money+profit
    
    if i<143:
        equal=pd.Series([1]*len(cangtop[i+1]),index=cangtop[i+1])
        target=equal*money*longratio/len(cangtop[i+1])
        exp=abs(target.sub(longcap,fill_value=0))
        exp=np.sum(exp)*(2/1000)/2 
    if i==143:
        exp=money*2/1000
    exp2=shortratio*money*(0.1/12+0.3/100)
    money=money-(exp+exp2)
    longm=longm+longpro-exp
    shortm=shortm+shortpro-exp2
    long.append(longm)
    short.append(shortm)
    low.append(money)
    #print(exp,exp2)
    #print(money,debt)

pr=(money)**(1/12)-1
print('年化收益率：',100*pr)
netval=pd.DataFrame(low,index=date,columns=['price'])
longprofit=pd.DataFrame(long,index=date,columns=['price'])
shortprofit=pd.DataFrame(short,index=date,columns=['price'])
#plt.plot(longprofit)
#plt.plot(shortprofit)
plt.plot(netval)
#%%
fig, ax1 = plt.subplots(dpi=800)
ax1.plot(netval,label='因子选股策略（左）',c='green')
ax1.set_xlabel('年份')
ax1.set_ylabel('资产净值')
ax2 = ax1.twinx()
ax2.plot(zhongzheng,label='基准表现（右）',c='red',alpha=0.7,linestyle='--')
ax2.set_ylabel('中证500指数')
ax2.set_ylim(0.8,2.5)
ax1.set_ylim(-1.5,20)
lines = [ax1.get_lines()[0], ax2.get_lines()[0]]
ax1.legend(lines, [line.get_label() for line in lines])

#plt.legend()
plt.title('50/50 多空组合')
plt.show()
#%%
fig, ax1 = plt.subplots(dpi=800)
ax1.plot(netval,label='因子选股',c='green')
ax1.set_xlabel('年份')
ax1.set_ylabel('资产净值')
ax1.plot(zhongzheng,label='基准表现',c='red',alpha=0.7,linestyle='--')
#lines = [ax1.get_lines()[0], ax2.get_lines()[0]]
ax1.legend(lines, [line.get_label() for line in lines])

plt.legend()
plt.title('五因子综合选股')
plt.show()
#%%
returns=netval.pct_change()
marvo=zhongzheng.pct_change()
marvo.drop(['20100701'],inplace=True)
#marvo.index=date2
#zhongzheng.index=date
returns.drop(['20100701'],inplace=True)
returns.index=returns.index-timedelta(1)
marvo.index=marvo.index-timedelta(1)

cova=np.cov(returns.T,marvo.T)
beta=cova[0,1]/cova[1,1]
print('beta:',beta)

ann=((netval.iloc[-1][0]/1)**(1/12)-1)*100
baseann=((zhongzheng.iloc[-1][0]/1)**(1/12)-1)*100
alpha=ann-3-beta*(baseann-3)
print('alpha:',alpha)

ex=returns-marvo
sharpe=np.sqrt(12)*(np.mean(returns['price'])-0.03/12)/np.std(returns['price'])
print('Sharpe ratio:',sharpe)

ir=(np.sqrt(12)*(np.mean(ex['price'])))/np.std(ex['price'])
#plt.plot(netval/zhongzheng,label='Net value ratio')
#print('Information ratio:',ir)

ret=returns+1
baseret=marvo+1
shouyi=100*(ret.groupby(ret.index.year).prod()-1)
baseshouyi=100*(baseret.groupby(baseret.index.year).prod()-1)
std=100*ret.groupby(ret.index.year).std()
std2=100*ex.groupby(ex.index.year).std()*np.sqrt(12)
std.iloc[0]*=np.sqrt(6)
std.iloc[-1]*=np.sqrt(6)
std.iloc[1:-1]*=np.sqrt(12)
excess=100*(((100+shouyi)/(100+baseshouyi))-1)

IR=0.01*excess/(np.sqrt(12)*ex.groupby(ex.index.year).std())
win=ex.groupby(ex.index.year).apply(lambda x: (x > 0).mean() * 100)
draw=100*returns.groupby(returns.index.year).apply(max_drawdown)
finalresult=pd.DataFrame({'年化收益率': shouyi.iloc[:,0],\
  '年化波动率': std2.iloc[:,0],'超额收益率': excess.iloc[:,0],\
      '月度胜率':win.iloc[:,0],'信息比率':IR.iloc[:,0],\
          '最大回撤（%）':draw.iloc[:,0]},index=shouyi.index
       )
print(finalresult.round(1).iloc[1:])

table = finalresult.round(1).iloc[1:].style.format(precision=1).set_table_styles([
    {'selector': 'thead th', 'props': [('text-align', 'center')]},
    {'selector': 'tbody td', 'props': [('text-align', 'center')]}
])
html=table.to_html()
with open('output.html', 'w') as f:
    f.write(html)


#%%
plt.scatter(returns,marvo)
#%%
year='2019'
plt.plot(netval[year]/zhongzheng[year],label='Top 2.5%')
#plt.plot(zhongzheng[year],label='Base')
plt.legend()
plt.title('50/50 Long-short')
plt.show()


