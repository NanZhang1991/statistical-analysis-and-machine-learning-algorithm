# -*- coding: utf-8 -*-



import pandas as pd
from scipy import stats
import re 
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

def single_sample_statistics(dataSer):
    result = pd.DataFrame([['N','均值','标准差','均值的标准误', ],
                           [dataSer.shape[0], dataSer.mean(), dataSer.std(),stats.sem(dataSer)],
                           [None, None, None, None]],index=['单样本统计量', dataSer.name, None])
    return result
    
def single_sample_test(dataSer, pop_mean, confidence=0.95):
    #t值 双尾p值
    t, p_twoTail=stats.ttest_1samp(dataSer, pop_mean)    
    # 自由度
    df = dataSer.shape[0]-1
    alpha = 1 - confidence
    t_score = stats.t.isf(alpha / 2, df)  
    # 均值差值
    d_mean = dataSer.mean() - pop_mean
    #差异指标 ：Cohen's d
    d= d_mean /dataSer.std()
    #置信区间
    ME = t_score * stats.sem(dataSer)
    lower_limit = dataSer.mean() - ME
    upper_limit = dataSer.mean() + ME
    result = pd.DataFrame([['检验值','t', 'df', 'Sig(双侧)', '均值差值', '差异指标', '置信水平', '置信区间下限', '置信区间上限'], 
                           [pop_mean, t, df, p_twoTail, d_mean, d, confidence, lower_limit, upper_limit]],index =['单样本检验', dataSer.name])    
    return result                        
  
#差异指标 ：Cohen's d
'''
Cohen's d 值介于0~1之间，该值越大说明差异幅度越大;
0 < Cohen's d <=0.2时，说明效应较小(差异幅度较小);


0.2 < Cohen's d <=0.8时，即 0.5附近时，说明效应中等(差异幅度中等);
Cohen's d > 0.8时，说明效应较大(差异幅度较大)。
'''
def get_parameter():
    print ("length:{}, content：{}".format(len(argv),argv))
    file_path = argv[argv.index('--file_path')+1]
    pop_mean = int(argv[argv.index('--pop_mean')+1])     
    if "--confidence" not in argv:        
        confidence = 0.95
    else:
        confidence = float("%.2f"%float(argv[argv.index('--confidence')+1])) 
    return file_path, pop_mean, confidence
        
def main(file_path, pop_mean, confidence):
#    df = pd.read_csv(file_path)
    df = get_data_hdfs(file_path)   
    dataSer = df.iloc[:,0]
    print(dataSer.name)
    single_sample_statistics_res = single_sample_statistics(dataSer)
    single_sample_test_res = single_sample_test(dataSer,pop_mean,confidence)
    res = pd.concat([single_sample_statistics_res, single_sample_test_res])
    print(res)
    res.to_csv('output/'+ re.findall('([^/]*).csv', file_path)[0] + '_res.csv', header=False)
    return res


if __name__=="__main__":
#    data = pd.DataFrame({'test':[15.6,16.2,22.5,20.5,16.4,19.4,16.6,17.9,12.7,13.9]})
#    file_path, pop_mean, confidence = './input/ssg_test.csv', 20, 0.95 #local test 
    
    file_path, pop_mean, confidence = get_parameter()
    res = main(file_path, pop_mean, confidence)
    cmd = "python single_sample_test.py --file_path ./input/ssg_test.csv --pop_mean 20 --confidence 0.95" 
    

