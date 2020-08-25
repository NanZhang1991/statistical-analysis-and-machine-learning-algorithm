# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 10:33:38 2020

@author: YJ001
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose



def draw_ts(timeSeries):
    f = plt.figure(facecolor='white')
    timeSeries.plot(color='blue')
    plt.show()


#单位根检验（Dickey-Fuller test）
def DFT(time_series):
    dftest = adfuller(time_series)  # ADF检验
  # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput

#差分序列的白噪声检验,若P值小于0.05，所以一阶差分后的序列为平稳非白噪声序列
def alx(time_series, lag=1):
    p_value = acorr_ljungbox(time_series, lags=lag)
    return p_value

# 自相关和偏相关图
def draw_acf_pacf(time_series):
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(time_series, ax=ax1)
    ax2 = f.add_subplot(212)
    plot_pacf(time_series, ax=ax2)
    plt.show()
    r,rac,Q = acf(time_series, qstat=True)
    prac = pacf(time_series,method='ywmle')
    table_data = np.c_[range(1,len(r)), r[1:],rac,prac[1:len(rac)+1],Q]
    table = pd.DataFrame(table_data, columns=['lag', "AC","Q", "PAC", "Prob(>Q)"])
    print(table)
    return table

# 平稳化处理
# a.对数变换
def ts_log(time_series):
    time_series = np.log(time_series)
    return(time_series)
#b.平滑法
# 移动平均图
def draw_trend(timeSeries, size):
    f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeSeries.rolling(window=size).mean()
    # 对size个数据进行加权移动平均
    rol_weighted_mean = pd.ewma(timeSeries, span=size)

    timeSeries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
    rol_weighted_mean.plot(color='black', label='Weighted Rolling Mean')
    plt.legend(loc='best')
    plt.title('Rolling Mean')
    plt.show()
    return rol_mean, rol_weighted_mean    
    
#c. 差分    
def ts_diff(time_series,d):
    time_series = time_series.diff(d).dropna(how=any)
    return(time_series)

#d. 分解
#所谓分解就是将时序数据分离成不同的成分。statime_seriesmodels使用的X-11分解过程，它主要将时序数据分离成长期趋势、季节趋势和随机成分。
#与其它统计软件一样，statime_seriesmodels也支持两类分解模型，加法模型和乘法模型，这里我只实现加法，乘法只需将model的参数设置为"multiplicative"即可。
def ts_decomposition(time_series_log):
    decomposition = seasonal_decompose(time_series_log, model="additive")
    
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid 
    return trend, seasonal, residual

#定阶
def get_pq():
    pmax = int(len(ts_diff)/10) #一般阶数不超过length/10
    qmax = int(len(ts_diff)/10) #一般阶数不超过length/10
    bic_matrix = [] #bic矩阵
    for p in range(pmax+1):
        tmp = []
        for q in range(qmax+1):
            try: 
                tmp.append(ARIMA(ts_diff, (p,1,q)).fit().bic)
            except:
                tmp.append(None)
        bic_matrix.append(tmp)
    
    bic_matrix = pd.DataFrame(bic_matrix) #从中可以找出最小值
    
    p,q = bic_matrix.stack().idxmin() 
    print(u'BIC最小的p值和q值为：%s、%s' %(p,q))
    return p,q



def ts_arma(ts, p, q):
    arma = ARMA(ts, order=(p, q)).fit(disp = -1)
    ts_predict_arma = arma.predict()
    print(arma.summary())
    return ts_predict_arma
    
# 加入差分的 ARMA 模型
def ts_arima(ts,p,d,q,start,end):
    arima = ARIMA(ts,(p,d,q,start,end)).fit(disp=-1,maxiter=100)
    print(arima.summary())
    ts_predict_arima = arima.predict(start,end, dynamic = False)
    return ts_predict_arima

if __name__ =="__main__":
    time_series = pd.Series([151.0, 188.46, 199.38, 219.75, 241.55, 262.58, 328.22, 
                         396.26, 442.04, 517.77, 626.52, 717.08, 824.38, 913.38,
                         1088.39, 1325.83, 1700.92, 2109.38, 2499.77, 2856.47, 
                         3114.02, 3229.29, 3545.39, 3880.53, 4212.82, 4757.45, 
                         5633.24, 6590.19, 7617.47, 9333.4, 11328.92, 12961.1, 15967.61])
    time_series.index = pd.Index(pd.period_range(1978,2010,freq='Y'))
