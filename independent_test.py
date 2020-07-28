# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:39:41 2020

@author: YJ001
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.stats.weightstats as st
from sys import argv
from hdfs.client import Client


def get_data_hdfs(file_path):    
    HDFSUrl = "http://192.168.0.201:50070"
    client = Client(HDFSUrl, root='/')
    with client.read(file_path, buffer_size=1024, delimiter='\n', encoding='utf-8') as reader:
        data = [line.strip().split() for line in reader]
        print("data",data[0:5])
    df = pd.DataFrame(data[1:],columns=data[0])
    return df

def group_statistics(x, y):      
    result = pd.DataFrame([['N', '均值', '标准差', '均值的标准误'], 
                           [x.shape[0], x.mean(), x.std(), stats.sem(x)],
                           [y.shape[0], y.mean(), y.std(), stats.sem(y)],
                           [None, None, None, None]], index=['组统计量', x.name, y.name, None])
    return result

def independent_sample_test(x, y, confidence=0.95):
    #方差齐性检验
    F, Sig = stats.levene(x, y)
    #t值 双尾p值
    t, p_two, df = st.ttest_ind(x,y, usevar='pooled')    
    alpha = 1 - confidence
    t_score = stats.t.isf(alpha / 2, df)  
    #样本均值差值
    sample_mean = x.mean() -  y.mean()
    #合并标准差
    sp = np.sqrt(((x.shape[0]-1)*np.square(x.std()) + (y.shape[0]-1)* np.square(y.std()))\
               / (x.shape[0] + y.shape[0] -2))    
    #效应量Cohen's d
    d = sample_mean / sp    
    #置信区间
    SE = np.sqrt(np.square(x)/x.shape[0] + np.square(y)/y.shape[0] )  
    lower_limit = sample_mean - t_score * SE
    upper_limit = sample_mean + t_score * SE
    
    t2 , p_two2, df2 = st.ttest_ind(x,y, usevar='unequal')   
    t_score2 = stats.t.isf(alpha / 2, df2)  
    lower_limit2 = sample_mean -  t_score2 * SE
    upper_limit2 = sample_mean +  t_score2 * SE   
    
    result = pd.DataFrame([['F', 'Sig' ,'t', 'df', 'Sig(双侧)','效应量','均值差值','置信水平', '置信区间下限', '置信区间上限'],
                           [F,  Sig, t, df, p_two, d, sample_mean,  confidence, lower_limit, upper_limit],
                           [None, None, t2, df2, p_two2, d, sample_mean, confidence, lower_limit2, upper_limit2]],
                           index=['独立样本检验', '假设方差相等', '假设方差不相等'])    
    return result 

def get_parameter():
    print ("length:{}, content：{}".format(len(argv),argv))
    file_path = argv[argv.index('--file_path')+1]
    try:
        df = get_data_hdfs(file_path)
    except Exception as e:
        print(e,'Can not get data from hdfs, use test data from local' )
        df = pd.read_csv(file_path)  #  
    if "--confidence" not in argv:        
        confidence = 0.95
    else:
        confidence = float("%.2f"%float(argv[argv.index('--confidence')+1])) 
    return df, confidence

def main(df, confidence, outfilename='ind'):  
    x = df.iloc[:,0]
    y = df.iloc[:,1]
    gs_res = group_statistics(x,y)
    ist_res = independent_sample_test(x, y, confidence)
    res = pd.concat([gs_res, ist_res])
    print(res)
    res.to_csv('test_data/output/' + outfilename + '_res.csv', header=False)
    return res


if __name__=="__main__":
    #local test 
#    df = pd.DataFrame({'x':[20.5, 19.8, 19.7, 20.4, 20.1, 20.0, 19.0, 19.9],'y':[20.7, 19.8, 19.5, 20.8, 20.4, 19.6, 20.2, None]})
#    confidence =  0.95     
#    res = main(df, confidence)
    
    df, confidence = get_parameter()
    res = main(df, confidence)
    cmd = "python independent_test.py --file_path ./test_data/input/independent_test.csv --confidence 0.95" 