# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 10:50:40 2020

@author: YJ001
"""
import pandas as pd
import statsmodels.formula.api as smf
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


def quantile_reg(df, quantile):
    mod = smf.quantreg(df.columns[1] + '~' + df.columns[0], df)
    res = mod.fit(q = quantile)
    print(res.summary())
    get_y = lambda a, b: a + b * df.iloc[:,0].values
    pre_y = get_y(res.params[0], res.params[1])
    qre_df = pd.DataFrame(data = [[quantile, res.params[0], res.params[1], pre_y,] + res.conf_int().iloc[1].tolist()], 
                               index = ['quantile_reg'], 
                               columns = ['qt','intercept','x_coef','cf_lower_bound','cf_upper_bound','pre_y'])
    return qre_df 

def get_parameter():
    print ("length:{}, content：{}".format(len(argv),argv))
    file_path = argv[argv.index('--file_path')+1]
    try:
        df = get_data_hdfs(file_path)
    except Exception as e:
        print(e,'Can not get data from hdfs, use test data from local' )
        df = pd.read_csv(file_path)  #  
    if "--quantile" not in argv:        
        quantile = 0.5
    else:
        quantile = float("%.1f"%float(argv[argv.index('--quantile')+1])) 
    return df, quantile

def main(df, quantile, outfilename='qrg'):  
    res = quantile_reg(df, quantile)
    print(res)
    res.to_csv('test_data/output/' + outfilename + '_res.csv', header=False)
    return res



if __name__=="__main__":
    #local test 
#    df = pd.DataFrame({'x':[20.5, 19.8, 19.7, 20.4, 20.1, 20.0, 19.0, 19.9],'y':[20.7, 19.8, 19.5, 20.8, 20.4, 19.6, 20.2, 20.3]})
#    quantile =.5
#    res = main(df, quantile)
    
    df, quantile = get_parameter()
    res = main(df, quantile)
    cmd = "python quantile_reg.py --file_path ./test_data/input/qrg_test.csv --quantile 0.5"