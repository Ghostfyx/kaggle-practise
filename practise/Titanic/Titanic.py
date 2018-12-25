#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2018/12/24 下午10:48
@Author  : fanyuexiang
@Site    : 
@File    : Titanic.py
@Software: PyCharm
@version: 1.0
@describe:
'''
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# pandas一些属性设置
pd.set_option('max_colwidth',200)
pd.set_option('display.width',200)
pd.set_option('display.max_columns',500)
pd.set_option('display.max_rows',1000)

print('read data begin')
train_file_path = '../../data/Titanic/train.csv'
test_file_path = '../../data/Titanic/test.csv'
train_data = pd.read_csv(train_file_path)
print(train_data.columns)
print(train_data.head(1))
print(train_data.info())