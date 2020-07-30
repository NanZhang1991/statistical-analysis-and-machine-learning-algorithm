# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 10:27:05 2020

@author: YJ001
"""
import csv
import pandas as pd

file_path = 'C:/Users/YJ001/Desktop/project/algorithm/test_data/input/independent_test.csv' 
with open(file_path,'r', encoding='utf-8') as f:    
     reader = csv.reader(f)   
     data = [line for line in reader]
     print(data)
df = pd.DataFrame(data[1:],columns=data[0])         


file_path = 'C:/Users/YJ001/Desktop/project/algorithm/test_data/input/independent_test.csv' 
with open(file_path,'r', encoding='utf-8') as f:    
    L= [line.strip().split(',')for line in f.readlines()]
    print(L)

#df = pd.DataFrame(L[1:],columns=L[0])
         