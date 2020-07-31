# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 13:09:55 2020

@author: YJ001
"""
from sys import argv
from hdfs.client import Client
import pandas as pd
import numpy as np
from factor_analyzer import factor_analyzer, Rotator
import numpy.linalg as nlg
import scipy.stats as st
import math

''''''
def kmo(df):
    # 相关系数
    corr = df.corr().values
    # print corr
    # 逆矩阵
    corr_inv = np.linalg.inv(corr)
    # 对角线取倒数，其他为0
    S2 = np.diag(corr_inv)
    S2 = np.linalg.inv(np.diag(S2))
    # 反映像协方差矩阵
    AIS = np.dot(S2, corr_inv)
    AIS = np.dot(AIS, S2)
    # 是映像协方差矩阵
    IS = corr + AIS - 2 * S2
    # 将矩阵AIS对角线上的元素开平方，并且将其余元素都变成0
    Dai = np.diag(np.sqrt(np.diag(AIS)))
    # IR是映像相关矩阵
    IR = np.dot(np.linalg.inv(Dai), IS)
    IR = np.dot(IR, np.linalg.inv(Dai))

    # AIR是反映像相关矩阵
    AIR = np.dot(np.linalg.inv(Dai), AIS)
    AIR = np.dot(AIR, np.linalg.inv(Dai))
    a = np.power((AIR - np.diag(np.diag(AIR))), 2)
    a = np.sum(a, axis=1)
    AA = np.sum(a)

    b = corr - np.identity(15)
    b = np.power(b, 2)
    b = np.sum(b, axis=1)
    BB = np.sum(b)

    MSA = b / (b + a)
    AIR = AIR - np.identity(15) + np.diag(MSA)

    kmo = BB / (AA + BB)
    return kmo

def bart(df):
    corr = df.corr().values
    # 计算结果有问题
    # bart=st.bartlett(*corr)
    detCorr = np.linalg.det(corr)
    n = len(df)
    p = len(df.columns)
    statistic = -math.log(detCorr) * (n - 1 - (2 * p + 5) / 6)
    df = p * (p - 1) / 2
    # 双侧概率
    pval = (1.0 - st.chi2.cdf(statistic, df)) * 2
    return statistic,pval
''''''

def get_data_hdfs(file_path):    
    HDFSUrl = "http://192.168.0.201:50070"
    client = Client(HDFSUrl, root='/')
    with client.read(file_path, buffer_size=1024, delimiter='\n', encoding='utf-8') as reader:
        data = [line.strip().split() for line in reader]
        print("data",data[0:5])
    df = pd.DataFrame(data[1:],columns=data[0])
    return df

def out_corr(df2):
    # 皮尔森相关系数
    df2_corr = df2.corr() 
    
    out_df2_corr_name = pd.DataFrame([[None],['皮尔森相关系数']])
    out_df2_corr_header = pd.DataFrame([df2.columns])
    out_df2_corr_data = pd.DataFrame(df2_corr.values)
    out_df2_corr = pd.concat([out_df2_corr_name, out_df2_corr_header, out_df2_corr_data])
    out_df2_corr_dum_index = [None,None,None]
    out_df2_corr_dum_index.extend(df2.columns)
    out_df2_corr.insert(0,'index', out_df2_corr_dum_index)
    out_df2_corr.columns = np.arange(out_df2_corr.shape[1])
#    print(out_df2_corr)    
    return out_df2_corr
    
def out_Ade(df2):
    #充分性测试(Adequacy Test)
    #KMO值：0.9以上非常好；0.8以上好；0.7一般；0.6差；0.5很差；0.5以下不能接受；巴特利球形检验的值范围在0-1，越接近1，使用因子分析效果越好
    #Kaiser-Meyer-Olkin Test
    kmo_all,kmo_model = factor_analyzer.calculate_kmo(df2)
    print('kmo:{}'.format(kmo_model))
    #Bartlett's Test
    chi_square_value,p_value = factor_analyzer.calculate_bartlett_sphericity(df2)
    print('Bartlett_p:{}'.format(p_value))
    out_Ade_df = pd.DataFrame([['充分性测试(Adequacy Test)',None],['kmo','Bartlett_p'],[kmo_model, p_value]])
#    print(out_Ade_df)
    return out_Ade_df

    
def _out_eig(df2):
    # 求特征值和特征向量
    eig_value, eigvector = nlg.eig(df2.corr())  # 求矩阵R的全部特征值，构成向量
    eig = pd.DataFrame({'names':df2.columns, 'eig_value':eig_value})
    eig.sort_values('eig_value', ascending=False, inplace=True)
    print("\n特征值\n：",eig)
    eig1=pd.DataFrame(eigvector,columns = df2.columns, index = df2.columns)
    print("\n特征向量\n",eig1)   
#    print("\n输出的特征向量\n",out_eig1)    
    out_eig_name = pd.DataFrame([[None],['特征值']])
    out_eig_data = pd.DataFrame({0:df2.columns, 1:eig_value})
    out_eig_data.sort_values(1, ascending=False, inplace=True)
    out_eig_header = pd.DataFrame({0:['names'],1:['eig_values']})
    out_eig = pd.concat([out_eig_name, out_eig_header,out_eig_data]) 
#    print("\n输出的特征值\n：",out_eig)
    out_eigl_name = pd.DataFrame([[None],['特征向量']])
    out_eig1_header = pd.DataFrame([df2.columns])
    out_eig1_data = pd.DataFrame(eigvector) 
    out_eig1 = pd.concat([out_eigl_name , out_eig1_header, out_eig1_data])
    out_eig_dum_index = [None,None,None]
    out_eig_dum_index.extend(df2.columns)
    out_eig1.insert(0,'index', out_eig_dum_index)
    out_eig1.columns = np.arange(out_eig1.shape[1])    
#    print("\n输出的特征向量\n",out_eig1)
    return eig, eig1, out_eig, out_eig1


def _out_a(df2):
    eig_value, eigvector = nlg.eig(df2.corr())  # 求矩阵R的全部特征值，构成向量
    eig = pd.DataFrame({'names':df2.columns, 'eig_value':eig_value})
    eig.sort_values('eig_value', ascending=False, inplace=True)
    # 求公因子个数m,使用前m个特征值的比重大于85%的标准，选出了公共因子是五个
    for m in range(1, df2.shape[1]):
        if eig['eig_value'][:m].sum() / eig['eig_value'].sum() >= 0.85:
            print("\n公因子个数:", m)
            break
            # 因子载荷阵
    A = np.mat(np.zeros((df2.shape[1], m)))
    i = 0
    j = 0
    while i < m:
        j = 0
        while j < df2.shape[1]:
            A[j:, i] = np.sqrt(eig_value[i]) * eigvector[j, i]
            j = j + 1
        i = i + 1
    a_columns = []
    for i in range(m):
        a_column = 'factor'+str(i+1)
        a_columns.append(a_column)
    a = pd.DataFrame(A, columns=a_columns, index=df2.columns)
    print("\n因子载荷阵\n", a)
    
    out_a_name = pd.DataFrame([[None],['因子载荷阵']])
    out_a_header = pd.DataFrame([a_columns])
    out_a_data = pd.DataFrame(A)
    out_a = pd.concat([out_a_name, out_a_header, out_a_data])
    out_a_dum_index = [None,None,None]
    out_a_dum_index.extend(df2.columns)
    out_a.insert(0,'index', out_a_dum_index)
    out_a.columns = np.arange(out_a.shape[1])
    return m, a, out_a      

def _out_cfv_var(df2, a_columns, m):
#    print(out_a)
    # 主成分提取参数设置:
    #  n_factors = 6:56个主成分
    #  method = 'principal'：提取方法为主成分提取方法
    #  rotation = 'varimax'：旋转方法为最大方差法
    fa = factor_analyzer.FactorAnalyzer(n_factors=m, method='principal',rotation='varimax')
    fa.fit(df2)
#    fa.loadings_ = a
    cfv = fa.get_communalities()
    print ("特殊因子方差:\n", cfv)
    
    out_cfv_name = pd.DataFrame([[None],['特殊因子方差']])
    out_cfv_data = pd.DataFrame(cfv)
    out_cfv = pd.concat([out_cfv_name, out_cfv_data])
    out_cfv_dum_index = [None,None]
    out_cfv_dum_index.extend(df2.columns)
    out_cfv.insert(0,'index', out_cfv_dum_index)
    out_cfv.columns = np.arange(out_cfv.shape[1]) 
#    print(out_cfv)
    
    var = fa.get_factor_variance()
    print ("\n解释的总方差（即贡献率):\n", var)
    out_var_name = pd.DataFrame([[None],['解释的总方差（即贡献率']])
    out_var_header = pd.DataFrame([a_columns])
    out_var_data = pd.DataFrame(np.array(var))
    out_var = pd.concat([out_var_name, out_var_header, out_var_data])
#    print(out_var)
    return cfv, var, out_cfv ,out_var

def _out_load_matrix(df2, a_columns, m):
    fa = factor_analyzer.FactorAnalyzer(n_factors=m, method='principal',rotation='varimax')
    fa.fit(df2)
    # # 旋转后的因子载荷矩阵
    rotator = Rotator()
    load_matrix = pd.DataFrame(rotator.fit_transform(fa.loadings_), columns = a_columns, index = df2.columns)
    print("\n因子旋转:\n", load_matrix)  
    
    out_load_matrix_name = pd.DataFrame([[None],['因子旋转']])
    out_load_matrix_header = pd.DataFrame([a_columns])
    out_load_matrix_data = pd.DataFrame(rotator.fit_transform(fa.loadings_))
    out_load_matrix = pd.concat([out_load_matrix_name, out_load_matrix_header, out_load_matrix_data])
    out_load_matrix_dum_index = [None,None,None]
    out_load_matrix_dum_index.extend(df2.columns)
    out_load_matrix.insert(0,'index', out_load_matrix_dum_index)
    out_load_matrix.columns = np.arange(out_load_matrix.shape[1]) 
    print(out_load_matrix)
    return load_matrix, out_load_matrix

def out_fs_fts(df2, load_matrix, a_columns):
    # 因子得分
    corr = np.matrix(df2.corr())
    load_matrix = np.matrix(load_matrix)
    factor_score = np.dot(np.linalg.inv(corr),load_matrix)
    fs = pd.DataFrame(factor_score, index=df2.columns, columns=a_columns)
    print("\n因子得分：\n", fs)
    
    out_fs_name = pd.DataFrame([[None],['因子得分']])
    out_fs_header = pd.DataFrame([a_columns])
    out_fs_data = pd.DataFrame(factor_score)
    out_fs = pd.concat([out_fs_name, out_fs_header, out_fs_data])
    out_fs_dum_index = [None,None,None]
    out_fs_dum_index.extend(df2.columns)
    out_fs.insert(0,'index', out_fs_dum_index)
    out_fs.columns = np.arange(out_fs.shape[1])       
    print(out_fs)
    
    fa_t_score = np.dot(np.mat(df2), np.mat(factor_score))
    print("\n每个id对应的因子得分：\n",pd.DataFrame(fa_t_score))
    out_fts_name = pd.DataFrame([[None],['每个id对应的因子得分']])
    out_fts_header = pd.DataFrame([a_columns])
    out_fts_data = pd.DataFrame(fa_t_score)
    out_fts = pd.concat([out_fts_name, out_fts_header, out_fts_data])  
    print(out_fts)
    return fs, fa_t_score, out_fs, out_fts 

def get_Fac_score(df2,columns,n,fs):
    means = df2.mean()
    stds = df2.std()
    arr = fs.iloc[:,n]
    sum=0
    for column in columns:
        sum=sum+(df[column ]-means[column])/(stds[column])*arr[column]
    return sum    

def total_score(df,df2,m,var,fs):
    # 各主成分得分
    # F=(成分1*贡献率1+成分2*贡献率2+….+成分4*贡献率4)/累计贡献率：
    df3 = df2.copy()
    total = 0
    for n in range(m):
        df3['FAC'+ str(n+1)]= get_Fac_score(df3, df.columns,n,fs)
        total = total + df3['FAC'+ str(n+1)]*var[1][n]
    df3['FAC'] = total/var[2][m-1]  
        
    FAC = df3.apply(lambda x:(x['FAC1']*var[1][0]+x['FAC2']*var[1][1]+
               x['FAC3'] * var[1][2]+x['FAC4']*var[1][3]+x['FAC5']*var[1][4]+
               x['FAC6']*var[1][5])/var[2][5], axis=1)
    
    return df3

def get_parameter():
    print ("length:{}, content：{}".format(len(argv),argv))
    file_path = argv[argv.index('--file_path')+1]
    try:
        df = get_data_hdfs(file_path)
    except Exception as e:
        print(e,'Can not get data from hdfs, use test data from local' )
        df = pd.read_csv(file_path) 
    return df

def main(df):
    # 将原始数据标准化处理 
    df2 = (df-df.mean())/df.std()  
    out_df2_corr = out_corr(df2)
    out_Ade_df = out_Ade(df2)
    eig, eig1, out_eig, out_eig1 = _out_eig(df2)
    m, a, out_a  = _out_a(df2)  
    cfv, var, out_cfv ,out_var =  _out_cfv_var(df2, a.columns, m)
    load_matrix, out_load_matrix = _out_load_matrix(df2, a.columns, m)
    fs, fa_t_score, out_fs, out_fts = out_fs_fts(df2, load_matrix, a.columns)
    fac_total = pd.concat([out_df2_corr, out_Ade_df, out_eig, out_eig1, out_a, out_cfv ,out_var, out_load_matrix, out_fs])
    df3 = total_score(df,df2,m,var,fs)
    return fac_total, out_fts, df3
    
    
if __name__ == '__main__':
#    df=pd.read_csv("C:/Users/YJ001/Desktop/project/algorithm/test_data/input/fa_analysis.csv")
#    local test 
    df = get_parameter()
    fac_total, out_fts, df3 =  main(df)
    fac_total.to_csv('C:/Users/YJ001/Desktop/project/algorithm/test_data/output/fac_bas.csv',index=False)
    out_fts.to_csv('C:/Users/YJ001/Desktop/project/algorithm/test_data/output/fac_each.csv',index=False)
    df3.to_csv('C:/Users/YJ001/Desktop/project/algorithm/test_data/output/fac_tscore.csv',index=False) 
    
    cmd = "python fa_analysis.py --file_path C:/Users/YJ001/Desktop/project/algorithm/test_data/input/fa_analysis.csv"
