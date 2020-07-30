# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 13:09:55 2020

@author: YJ001
"""

import pandas as pd
import numpy as np
from factor_analyzer import factor_analyzer, Rotator
import numpy.linalg as nlg
import scipy.stats as st
import math

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


df=pd.read_csv("C:/Users/YJ001/Desktop/project/algorithm/test_data/input/fa_analysis.csv")

# 将原始数据标准化处理 
df2=(df-df.mean())/df.std()
# 皮尔森相关系数
df2_corr=df2.corr()

#充分性测试(Adequacy Test)
#Kaiser-Meyer-Olkin Test
kmo_all,kmo_model = factor_analyzer.calculate_kmo(df)
#Bartlett's Test
chi_square_value,p_value = factor_analyzer.calculate_bartlett_sphericity(df)

# 求特征值和特征向量
eig_value, eigvector = nlg.eig(df2_corr)  # 求矩阵R的全部特征值，构成向量
eig = pd.DataFrame()
eig['names'] = df2_corr.columns
eig['eig_value'] = eig_value
eig.sort_values('eig_value', ascending=False, inplace=True)
print("\n特征值\n：",eig)
eig1=pd.DataFrame(eigvector)
eig1.columns = df2_corr.columns
eig1.index = df2_corr.columns
print("\n特征向量\n",eig1)

# 求公因子个数m,使用前m个特征值的比重大于85%的标准，选出了公共因子是五个
for m in range(1, 15):
    if eig['eig_value'][:m].sum() / eig['eig_value'].sum() >= 0.85:
        print("\n公因子个数:", m)
        break
# 主成分提取参数设置:
#  n_factors = 5:5个主成分
#  method = 'principal'：提取方法为主成分提取方法
#  rotation = 'varimax'：旋转方法为最大方差法
fa = factor_analyzer.FactorAnalyzer(n_factors=m, method='principal',rotation='varimax')
fa.fit(df)
cfv = fa.get_communalities()
print ("公因子方差:\n", cfv)
var = fa.get_factor_variance()
print ("\n解释的总方差（即贡献率):\n", var)


# # 旋转后的因子载荷矩阵
rotator = Rotator()
load_matrix = rotator.fit_transform(fa.loadings_)

# 因子得分
corr = np.matrix(df2_corr)
load_matrix = np.matrix(load_matrix)
factor_score = np.dot(np.linalg.inv(corr),load_matrix)
factor_score = pd.DataFrame(factor_score)
factor_score.index = df2_corr.columns
print("\n因子得分：\n", factor_score)
fa_t_score = np.dot(np.mat(df2), np.mat(factor_score))
print("\n对应因子得分：\n",pd.DataFrame(fa_t_score))


# 各主成分得分
def get_Fac_score(df,columns,n):
    means = df.mean()
    stds = df.std()
    arr = factor_score[n]
    sum=0
    for column in columns:
        sum=sum+(df[column ]-means[column])/(stds[column])*arr[column]
    return sum

total = 0
# F=(成分1*贡献率1+成分2*贡献率2+….+成分4*贡献率4)/累计贡献率：
for n in range(m):
    i = n + 1
    print('FAC'+ str(i))
    columns = df.columns
    df2['FAC'+ str(i)]= get_Fac_score(df2,columns,n)
    total = total + df2['FAC'+ str(i)]*var[1][n]
FAC = total/var[2][5]  
    
df2['FAC'] = df2.apply(lambda x:(x['FAC1']*var[1][0]+x['FAC2']*var[1][1]+
           x['FAC3'] * var[1][2]+x['FAC4']*var[1][3]+x['FAC5']*var[1][4]+
           x['FAC6']*var[1][5])/var[2][5], axis=1)
df2.to_excel('result_fac.xls',index=False)

#if __name__ == '__main__':
