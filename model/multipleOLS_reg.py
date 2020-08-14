# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 08:51:34 2020

@author: YJ001
"""

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


def multipleOLS(df):
    x = df.iloc[:,0:-1]
    y = df.iloc[:,-1]
    X = sm.add_constant(x)#给自变量中加入常数项
    model = sm.OLS(y,X)
    res = model.fit()
    print(res.summary())
#    pre_y = res.predict(sm.add_constant(x))
    mOLS_df = pd.DataFrame({'x_coef':res.params,'cf_lower_bound':res.conf_int()[0],'cf_upper_bound':res.conf_int()[1]})
    return  mOLS_df

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
        df = pd.read_csv(file_path)    
    
    res = multipleOLS(df)
    print(res)
    if outpath != None:
        dataframe_write_to_hdfs(outpath, res)
    else:
        res.to_csv('C:/Users/YJ001/Desktop/project/algorithm/test_data/output/mOLS_res.csv',index=False)
    return res




if __name__=="__main__":

    file_path, outpath = get_parameter()
    res = main(file_path, outpath)
    
#    local test 
    cmd = "python multipleOLS_reg.py --file_path C:/Users/YJ001/Desktop/project/algorithm/test_data/input/multiple_OLS.csv"