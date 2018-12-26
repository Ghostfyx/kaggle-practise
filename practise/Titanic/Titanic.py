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
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt

# pandas一些属性设置
pd.set_option('max_colwidth',200)
pd.set_option('display.width',200)
pd.set_option('display.max_columns',500)
pd.set_option('display.max_rows',1000)

print('read data begin')
train_file_path = '../../data/Titanic/train.csv'
test_file_path = '../../data/Titanic/test.csv'
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)
print(train_data.columns)
print(train_data.head(1))
print('train_data_info:',train_data.info())
print('test_data_info:',test_data.info())
'''
train_data:Age,Embarked and Cabin have nan value
test_data:Age,Cabin,Fare have nan value
'''
print('begin data visualization')
# DataFrame.plot( )画图函数
train_data['Survived'].value_counts().plot(autopct = '%1.2f%%', kind='pie')
plt.show()
# 乘客各属性与结果之间的关系
print("分析性別与生存之间的关系")
train_data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()
print("")
plt.show()
print('begin handle missing data')
survived_data = train_data['Survived']
train_data = train_data.drop(['Survived'], axis=1)
data = pd.concat([train_data,test_data], ignore_index=True)
missing_data_count = data.isnull().sum().sort_values(ascending=False)
missing_data_rate = missing_data_count/len(data)
missing_data = pd.concat([missing_data_count, missing_data_rate], axis=1, keys=['count','rate'])
print(missing_data)
'''
处理缺失值的方法：
    1.age：根据Pclass，Sex，SibSp，Parch共同决定
    2.Cabin：根据Pclass，Fare决定
    3.Fare：Pclass，Cabin
    4.Embarked：使用众数代替
'''
data['Embarked'].fillna(data['Embarked'].mode(), inplace=True)
# 使用回归 随机森林等模型来预测缺失属性的值
age_df = [['Age','Fare', 'Parch', 'SibSp', 'Pclass']]





