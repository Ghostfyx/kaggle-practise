import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
import numpy as np

def analysis_train_data():
    file_path = '../../data/houseprice/train.csv'
    house_data_train = pd.read_csv(file_path)
    print("train data length----->"+str(len(house_data_train)))
    feature_list = house_data_train.columns
    print(feature_list)
    print(house_data_train)
    # 默认n=5,n表示查看的行数
    print(house_data_train.head(n=5))
    # 对训练数据的房子价格进行分析，计算平均值，标准化值等
    print(house_data_train["SalePrice"].describe())
    # 利用数据可视化绘制房子价格的直方图，根据直方图分析房价的数据分布
    # 发现房价数据正偏，偏离正太分布，有峰值
    plt.figure(figsize=(18.5, 10.5))
    sns.distplot(house_data_train['SalePrice'], kde=True, rug=True, hist=True)
    '''曲线的峰度(kurtosis)和偏度(skewness)。
    峰度(kurtosis)描述变量取值分布形态的陡缓程度的统计量。
    偏度(skewness)是描述变量取值分布对称性的统计量
        kurtosis=0与正态分布的陡缓程度相同。
        kurtosis>0比正态分布的高峰更加陡峭。
        kurtosis<0比正态分布的高峰平。
        skewness=0分布形态与正态分布偏度相同。
        skewness>0表示正(右)偏差数值较大，右边的尾巴比较长。
        skewness<0表示负(左)偏差数值较大，左边的尾巴比较长。
    '''
    print("Skewness: %f" % house_data_train['SalePrice'].skew())
    print("Kurtosis: %f" % house_data_train['SalePrice'].kurt())
    null_count = house_data_train.isnull().sum().sort_values(ascending=False)
    print(null_count)
    # 'OverallQual(总体材料和加工质量,装修)', 'GrLivArea(生活区平方英尺,住房面积)', 'YearBuilt(建筑年代)', 'TotalBsmtSF(地下室面积)' 是熟悉房价的人分析出的与房价密切相关的属性
    plt.figure(figsize=(18.5, 10.5))
    sns.pairplot(x_vars=['OverallQual', 'GrLivArea', 'YearBuilt', 'TotalBsmtSF'], y_vars=['SalePrice'], data=house_data_train)
    # 绘制上述变量与价格的箱型图
    data_OverallQual = pd.concat([house_data_train['SalePrice'], house_data_train["OverallQual"]], axis=1)
    data_GrLivArea = pd.concat([house_data_train['SalePrice'], house_data_train["GrLivArea"]], axis=1)
    data_YearBuilt = pd.concat([house_data_train['SalePrice'], house_data_train["YearBuilt"]], axis=1)
    data_TotalBsmtSF = pd.concat([house_data_train['SalePrice'], house_data_train["TotalBsmtSF"]], axis=1)
    # figure作用：新建画布
    plt.figure(figsize=(18.5, 10.5))
    plt.subplot(2,2,1)
    sns.boxplot(x="OverallQual", y="SalePrice", data=data_OverallQual)
    plt.subplot(2, 2, 2)
    sns.boxplot(x="GrLivArea", y="SalePrice", data=data_GrLivArea)
    plt.subplot(2, 2, 3)
    sns.boxplot(x="YearBuilt", y="SalePrice", data=data_YearBuilt)
    plt.subplot(2, 2, 4)
    sns.boxplot(x="TotalBsmtSF", y="SalePrice", data=data_TotalBsmtSF)
    # 绘制相关系数矩阵
    plt.figure(figsize=(18.5, 10.5))
    house_corr = house_data_train.corr()
    sns.heatmap(house_corr, vmax=0.8, square=True);
    # 绘制相关系数矩阵
    plt.figure(figsize=(18.5, 10.5))
    k = 10
    # nlargest的优点就是能一次看到最大的几行
    cols = house_corr.nlargest(k, columns='SalePrice')['SalePrice'].index
    cm = np.corrcoef(house_data_train[cols].values.T)
    sns.set(font_scale=1.25)
    sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                     yticklabels=cols.values, xticklabels=cols.values)
    plt.show()
    return house_data_train


def feature_handler(train_data):
    """
    第一步：对数据分析绘制的散点图异常数据的处理
    第二步：对缺省数据进行处理；处理方式：
           1.如果整列全部空值，删除整列
           2.如果部分为空值，采用众数替代
    第三步：对每一列的特征进行分析，分别将其放入离散类别特征；连续类标特征和数据类型特征
    对于类别特征，将其类别和数量输出到控制台，人工进行判断
    对于数据类型和类别特征中的缺省值和异常值都进行处理
    :param data_list:
    """
    train_data.drop(train_data[(train_data['OverallQual'] < 5) & (train_data['SalePrice'] > 200000)].index,
                    inplace=True)
    train_data.drop(train_data[(train_data['GrLivArea'] > 4000) & (train_data['SalePrice'] < 200000)].index,
                    inplace=True)
    train_data.drop(train_data[(train_data['YearBuilt'] < 1900) & (train_data['SalePrice'] > 400000)].index,
                    inplace=True)
    train_data.drop(train_data[(train_data['TotalBsmtSF'] > 6000) & (train_data['SalePrice'] < 200000)].index,
                    inplace=True)
    train_data.reset_index(drop=True, inplace=True)
    train_data.dropna(axis=1, how='any', inplace=True)
    # 第一步end
    contain_feature_list = []
    # 离散类别特征
    discrete_feature_list = []
    numeric_feature_list = []
    feature_map = {}
    for feature in train_data.columns[1:]:
        if train_data.dtypes[feature] == 'object':
            # 列的众数
            miss_feature_data = train_data[feature].mode()
            train_data.fillna(miss_feature_data, inplace=True)
            # # value_counts是pandas自带的统计值函数
            # feature_value_counts = train_data[feature].value_counts()
            # print(feature+'包含的类别个数为 ', len(feature_value_counts))
            # print(feature_value_counts.index)
            # label = input("请输入英文字母c或者d，c表虚连续类别，d表示离散类别：")
            # if label == 'c':
            #     contain_feature_list.appdend(feature)
            # elif label == 'd':
            discrete_feature_list.append(feature)
            # else:
            #     break
        elif train_data.dtypes[feature] != 'object':
            # 使用列平均值进行填充
            miss_feature_data = train_data[feature].mean()
            train_data.fillna(miss_feature_data)
            numeric_feature_list.append(feature)
    feature_map['contain_feature'] = contain_feature_list
    feature_map['discrete_feature'] = discrete_feature_list
    feature_map['numeric_feature'] = numeric_feature_list
    return feature_map


def feature_select(feature_map, data_list):
    """
    第一步，对训练集进行数据预处理，找出数据类型的特征和非数据类型的特征
    :param  data_list:训练数据集
    :param feature_map:特征类型集合
    """
    standard = StandardScaler()
    vec = LabelEncoder()
    for feature in feature_map['contain_feature']:
        a = data_list[feature].values
        data_list[feature] = vec.fit_transform(data_list[feature].values)
    for feature in feature_map['discrete_feature']:
        a = data_list[feature].values
        data_list[feature] = vec.fit_transform(data_list[feature].values)
    for feature in feature_map['numeric_feature']:
        # 注意此处reshape的作用：将数据转换为1列
        a = data_list[feature].values
        data_list[feature] = standard.fit_transform(data_list[feature].values.reshape(-1, 1))
    print("数据预处理完毕")
    pca = PCA(n_components =10, copy=True)
    data = data_list.iloc[:, :-1]
    data_list_pca = pca.fit(data)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_)
    return data_list_pca



if __name__ == '__main__':
    house_data_train = analysis_train_data()
    feature_map = feature_handler(house_data_train)
    feature_select(feature_map=feature_map, data_list=house_data_train)
