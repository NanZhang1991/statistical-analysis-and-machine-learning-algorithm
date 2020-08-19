# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 17:15:17 2020

@author: YJ001
"""
import pandas as pd 
import scipy.stats as stats
from sys import argv
from hdfs.client import Client



data = pd.DataFrame({'x':[20.5, 18.8, 19.8, 20.9, 21.5, 19.5, 21.0, 21.2],'y':[17.7, 20.3, 20.0, 18.8, 19.0, 20.1, 20.0, 19.1]})

def pair_sample_statistics(x, y):      
    result = pd.DataFrame([['N', '均值', '标准差', '均值的标准误'], 
                           [x.shape[0], x.mean(), x.std(), stats.sem(x)],
                           [y.shape[0], y.mean(), y.std(), stats.sem(y)],
                           [None, None, None, None]], index=['配对样本统计量', x.name, y.name, None])
    return result



def pair_sample_test(x, y,confidence=0.95):
    #t值 双尾p值
    t,p_twoTail = stats.ttest_rel(x,y)
    # 样本差值
    sample = x - y
    # 自由度
    df = sample.shape[0]-1
    alpha = 1 - confidence
    t_score = stats.t.isf(alpha / 2, df)  
    #置信区间
    ME = t_score * stats.sem(sample)
    lower_limit = sample.mean() - ME
    upper_limit = sample.mean() + ME
    #效应量
    d = sample.mean()/sample.std()
    result = pd.DataFrame([['均值', '标准差', '均值的标准误', '效应量','置信水平', '置信区间下限', '置信区间上限','t', 'df', 'Sig(双侧)'],
                           [sample.mean(),  sample.std(), stats.sem(sample), d, confidence, lower_limit, upper_limit, t, df, p_twoTail]],
                           index=['配对样本检验', '成对差分'])    
    return result  

def get_parameter():
    print ("length:{}, content：{}".format(len(argv),argv))
    file_path = argv[argv.index('--file_path')+1]     
    if "--confidence" not in argv:        
        confidence = 0.95
    else:
        confidence = float("%.2f"%float(argv[argv.index('--confidence')+1])) 
    if "--outpath" not in argv:
        outpath = None
    else:
        outpath = argv[argv.index('--outpath')+1]    
    return file_path, confidence, outpath

def get_data_hdfs(file_path):    
    HDFSUrl = "http://192.168.0.201:50070"
    client = Client(HDFSUrl, root='/')
    with client.read(file_path, buffer_size=1024, delimiter='\n', encoding='utf-8') as reader:
        data = [line.strip().split(',') for line in reader]
        print("data",data[0:5])
    df = pd.DataFrame(data[1:],columns=data[0])
    return df

def dataframe_write_to_hdfs(hdfs_path, dataframe):
    """
    :param client:
    :param hdfs_path:
    :param dataframe:
    :return:
    """
    HDFSUrl = "http://192.168.0.201:50070"
    client = Client(HDFSUrl, root='/')    
    client.write(hdfs_path, dataframe.to_csv(header=False,index=False,sep="\t"), encoding='utf-8',overwrite=True)
    
def main(file_path, confidence, outpath):
    try:
        df = get_data_hdfs(file_path)
    except Exception as e:
        print(e,'Can not get data from hdfs, use test data from local' )
        df = pd.read_csv(file_path)  #
    x = df.iloc[:,0]
    y = df.iloc[:,1]
    pair_sample_statistics_res = pair_sample_statistics(x,y)
    pair_sample_test_res = pair_sample_test(x, y, confidence)
    res = pd.concat([pair_sample_statistics_res, pair_sample_test_res])
    print(res)
    if outpath != None:
        dataframe_write_to_hdfs(outpath, res)
    else:
        res.to_csv('C:/Users/YJ001/Desktop/project/algorithm/test_data/output/pair_res.csv', header=False)
    return res

if __name__=="__main__":

    file_path, confidence, outpath = get_parameter()
    res = main(file_path, confidence, outpath)
    cmd = "python pair_test.py --file_path C:/Users/YJ001/Desktop/project/algorithm/test_data/input/pari_test.csv --confidence 0.95"     