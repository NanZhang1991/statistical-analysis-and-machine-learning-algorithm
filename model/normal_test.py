# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 13:56:09 2020

@author: YJ001
"""





import numpy as np
import pandas as pd
from scipy import stats
from hdfs.client import Client
from sys import argv


def normaltest(x):
    # kstest（K-S检验）
    K_s,K_p = stats.kstest(x, 'norm')
    print("kstest（K-S检验）",K_s,K_p)

    # Shapiro-Wilk test
    S_s,S_p = stats.shapiro(x)
    print('the shapiro test ',S_s,',',S_p)
    
    # normaltest
    N_s,N_p = stats.normaltest(x)
    print('normaltest',N_s,N_p)
    
    # Anderson-Darling test
    A_s,A_c,A_sl = stats.anderson(x,dist='norm')
    print('Anderson-Darling test',A_s,A_c,A_sl)

    df = x.shape[0]-1
    Norm_test = pd.DataFrame([['统计量', 'df', 'Sig'], 
                              [K_s, df, K_p],
                              [S_s, df, S_p],
                              [None, None, None]],
                              index=["正态性检验", 'kstest', 'Shapiro-Wilk', None])
    s_l = list(map(lambda x: '显著水平 '+str(x)+'%',A_sl))
    c_v = list(map(lambda x: '临界值 '+str(x),A_c))
    
    And_D_test = pd.DataFrame([['统计量', s_l[0], s_l[1], s_l[2], s_l[3], s_l[4]], 
                               [A_s, c_v[0],  c_v[1], c_v[2], c_v[3], c_v[3] ],
                               [None, None, None, None, None, None]],
                              index=["正态性检验", 'Anderson-Darling ', None])
    result = pd.concat([Norm_test, And_D_test])
    return result


def JBtest(x):
    # 样本规模n
    n = x.size
    x_ = x - x.mean()   
    
    """
    M2:二阶中心钜
    skew 偏度 = 三阶中心矩 与 M2^1.5的比
    krut 峰值 = 四阶中心钜 与 M2^2 的比
    """
    M2 = np.mean(x_**2)

    skew =  np.mean(x_**3)/M2**1.5

    krut = np.mean(x_**4)/M2**2

    """
    计算JB统计量，以及建立假设检验
    """
    JB_s = n*(skew**2/6 + (krut-3 )**2/24)
    JB_p = 1 - stats.chi2.cdf(JB_s, df=2)


    print("偏度：",stats.skew(x),skew)

    print("峰值：",stats.kurtosis(x)+3,krut)

    print("JB检验：",stats.jarque_bera(x))
    res = pd.DataFrame([['skew', 'krut', '统计量', 'Sig'], 
                        [skew, krut, JB_s, JB_p]],
                       index=["正态性检验", 'JB test'])
    return res

def get_parameter():
    print ("length:{}, content：{}".format(len(argv),argv))
    file_path = argv[argv.index('--file_path')+1]  
    if "--outpath" not in argv:
        outpath = None
    else:
        outpath = argv[argv.index('--outpath')+1]    
    return file_path, outpath

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
    
def main(file_path, outpath):   
    try:
        df = get_data_hdfs(file_path)
    except Exception as e:
        print(e,'Can not get data from hdfs, use test data from local' )
        df = pd.read_csv(file_path)  #    
    x = df.iloc[:,0]
    normaltest_res = normaltest(x)
    JBtest_res = JBtest(x)
    res = pd.concat([normaltest_res, JBtest_res])
    if outpath != None:
        dataframe_write_to_hdfs(outpath, res)
    else:
        res.to_csv('C:/Users/YJ001/Desktop/project/algorithm/test_data/output/nor_res.csv', header=False)
    print(res)
    return res
    
    
if __name__=="__main__":

    file_path, outpath = get_parameter()
    res = main(file_path, outpath)
    cmd = "python normal_test.py --file_path C:/Users/YJ001/Desktop/project/algorithm/test_data/input/normal_test.csv"    
    

 

