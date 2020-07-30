# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:30:46 2020

@author: YJ001
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 10:50:40 2020

@author: YJ001
"""
import pandas as pd
import statsmodels.api as sm
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


def unaryOLS(df):
    x = df.iloc[:,0]
    y = df.iloc[:,1]
    X = sm.add_constant(x)#给自变量中加入常数项
    model = sm.OLS(y,X)
    res = model.fit()
    print(res.summary())
#    get_y = lambda a, b: a + b * df.iloc[:,0].values
#    pre_y = get_y(res.params[0], res.params[1])
    uOLS_df = pd.DataFrame(data = [[res.params[0], res.params[1]] + res.conf_int().iloc[1].tolist()], 
                               index = ['uOLS_reg'], 
                               columns = ['intercept','x_coef','cf_lower_bound','cf_upper_bound']) 
    return uOLS_df

def get_parameter():
    print ("length:{}, content：{}".format(len(argv),argv))
    file_path = argv[argv.index('--file_path')+1]
    try:
        df = get_data_hdfs(file_path)
    except Exception as e:
        print(e,'Can not get data from hdfs, use test data from local' )
        df = pd.read_csv(file_path) 
    return df

def main(df, outfilename='uOLS'):  
    res = unaryOLS(df)
    print(res)
    res.to_csv('C:/Users/YJ001/Desktop/project/algorithm/test_data/output/' + outfilename + '_res.csv')
    return res


'''
Dep.Variable : 使用的参数值
Model：使用的模型
method：使用的方法
Data：时间
No.Observations：样本数据个数
Df Residuals：残差的自由度
DF Model：模型的自由度
R-squared：R方值
Adj.R-squared:调整后的R方
F-statistic :F统计量
Prob（F-statistic）：F统计量的p值
Log-Likelihood：似然度
AIC BIC：衡量模型优良度的指标，越小越好
const：截距项
P>|t| :t检验的p值，如果p值小于0.05 那么我们就认为变量是显著的
'''

if __name__=="__main__":

    
    df = get_parameter()
    res = main(df)
    
#    local test 
    cmd = "python unaryOLS_reg.py --file_path C:/Users/YJ001/Desktop/project/algorithm/test_data/input/unary_OLS.csv"