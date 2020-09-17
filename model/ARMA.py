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



def draw_ts(time_series):
    f = plt.figure(facecolor='white')
    time_series.plot(color='blue')
    plt.show()


#单位根检验（Dickey-Fuller test） 若P值大于0.05接受原假设存在单位根 不平稳
def DFT(time_series):
    dftest = adfuller(time_series)  # ADF检验
  # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput

#差分序列的白噪声检验,若P值小于0.05，则一阶差分后的序列为平稳非白噪声序列
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
def draw_trend(time_series, size):
    f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = time_series.rolling(window=size).mean()
    # 对size个数据进行加权移动平均
    rol_weighted_mean = pd.DataFrame.ewm(time_series, span=size).mean()

    time_series.plot(color='blue', label='Original')
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
def get_pq(ts_diff):
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



def ts_arma(ts, p, q, start,end):
    arma = ARMA(ts, order=(p, q)).fit(disp = -1)
    print("未来五年:", arma.forecast(5)[0])
    ts_predict_arma = arma.predict(start,end)
    print(arma.summary())
    return ts_predict_arma
    
# 加入差分的 ARMA 模型
def ts_arima(ts,p,d,q,start,end):
    arima = ARIMA(ts, order=(p,d,q)).fit(disp=-1,maxiter=100)
    print("未来五年:", arima.forecast(5)[0])
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
    draw_ts(time_series)
    dfoutput = DFT(time_series)
    w_p = alx(time_series, lag=1)
    ts_l = ts_log(time_series)
    draw_trend(ts_l, 12)
    #从上图可以发现窗口为12的移动平均能较好的剔除年周期性因素，而指数平均法是对周期内的数据进行了加权，
    #能在一定程度上减小年周期因素，但并不能完全剔除，如要完全剔除可以进一步进行差分操作。
    draw_ts(ts_l)
    
    diff_12 = ts_diff(ts_l,12)
    diff_12_12 = ts_diff(diff_12, 12)
    dfoutput2 = DFT(diff_12_12)
    p,q = get_pq(time_series)

#对于年周期成分我们使用窗口为12的移动平进行处理，
#对于长期趋势成分我们采用1阶差分来进行处理。    
    rol_mean = ts_l.rolling(window=3).mean()
    rol_mean.dropna(inplace=True)
    ts_diff_1 = rol_mean.diff(1)
    ts_diff_1.dropna(inplace=True)
    dfoutput3 = DFT(ts_diff_1) 
    
    ts_diff_2 = ts_diff_1.diff(1)
    ts_diff_2.dropna(inplace=True)
    dfoutput4 = DFT(ts_diff_2) 
    
    ts_predict_arma = ARMA(ts_diff_2, order=(p, q)).fit(disp = -1)
    predict_ts = ts_predict_arma.predict()
    # 一阶差分还原
    diff_shift_ts = ts_diff_1.shift(1)
    diff_recover_1 = predict_ts.add(diff_shift_ts)
    # 再次一阶差分还原
    rol_shift_ts = rol_mean.shift(1)
    diff_recover = diff_recover_1.add(rol_shift_ts)
    # 移动平均还原
    rol_sum = ts_l.rolling(window=2).sum()
    rol_recover = diff_recover*3 - rol_sum.shift(1)
    # 对数还原
    log_recover = np.exp(rol_recover)
    log_recover.dropna(inplace=True)
    
    #均方根误差（RMSE）来评估模型样本内拟合的好坏。利用该准则进行判别时，需要剔除“非预测”数据的影响。
    ts = time_series[log_recover.index]  # 过滤没有预测的记录
    plt.figure(facecolor='white')
    log_recover.plot(color='blue', label='Predict')
    ts.plot(color='red', label='Original')
    plt.legend(loc='best')
    plt.title('RMSE: %.4f'% np.sqrt(sum((log_recover-ts)**2)/ts.size))
    plt.show()

    d = 0
    ts_predict_arima = ts_arima(time_series,p,d,q,'1980', '1985')