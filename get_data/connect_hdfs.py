

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 17:14:59 2020

@author: YJ001
"""
import pandas as pd
from hdfs.client import Client


def get_data(file_path):    
    HDFSUrl = "http://192.168.0.201:50070"
    client = Client(HDFSUrl, root='/')
    with client.read(file_path, buffer_size=1024, delimiter='\n', encoding='utf-8') as reader:
        data = [line.strip().split() for line in reader]
        print("data",data[0:2])
    return data

file_path = '/EML/Data/FB2A21FC-278F-4FDE-8A1F-5021D0EF17AC/FB2A21FC-278F-4FDE-8A1F-5021D0EF17AC'
data = get_data(file_path)
df = pd.DataFrame(data[1:],columns=data[0])   




