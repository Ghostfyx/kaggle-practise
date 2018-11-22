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

top_feature_list = []
file_path = '../../data/houseprice/train.csv'
house_data_train = pd.read_csv(file_path)

def analysis_train_data():
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
    # 绘制相关系数矩阵
    plt.figure(figsize=(18.5, 10.5))
    house_corr = house_data_train.corr()
    # 得到与SalePrice相关性top10的特征属性
    corr_feature_sort = house_corr['SalePrice'].sort_values(ascending=False)
    top_feature_list = list(corr_feature_sort[(corr_feature_sort.values > 0.5) & (corr_feature_sort.values < 1)].index)
    # 'OverallQual,装修与房子质量', 'GrLivArea,住房面积', 'GarageCars,车库的车容量', 'GarageArea,车库大小', 'TotalBsmtSF,地下室面积',
    # '1stFlrSF,一楼面积', 'FullBath，全身的浴室数量', 'TotRmsAbvGrd,房间总数（不含浴室）', 'YearBuilt,建筑年代', 'YearRemodAdd,重构日期'
    print(top_feature_list)
    # vmin, vmax : 显示的数据值的最大和最小的范围
    sns.heatmap(house_corr, vmax=1, square=True)
    # 绘制相关系数矩阵
    plt.figure(figsize=(18.5, 10.5))
    k = len(top_feature_list)
    # nlargest的优点就是能一次看到最大的几行
    cols = house_corr.nlargest(k, columns='SalePrice')['SalePrice'].index
    cm = np.corrcoef(house_data_train[cols].values.T)
    sns.set(font_scale=1.25)
    sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                yticklabels=cols.values, xticklabels=cols.values)
    # 绘制散点图
    plt.figure(figsize=(18.5, 10.5))
    # sns.pairplot(x_vars=top_feature_list, y_vars=['SalePrice'], data=house_data_train)
    # enumerate同时遍历索引和内容
    for index, feature in enumerate(top_feature_list):
        plt.subplot(3, 4, index+1)
        sns.stripplot(x=feature, y="SalePrice", data=house_data_train)
    # figure作用：新建画布
    plt.figure(figsize=(18.5, 10.5))
    # 绘制上述变量与价格的箱型图
    for index, feature in enumerate(top_feature_list):
        data = pd.concat([house_data_train['SalePrice'], house_data_train[feature]], axis=1)
        plt.subplot(3, 4, index+1)
        sns.boxplot(x=feature, y="SalePrice", data=data)
    plt.show()

def feature_handler():
    """
    第一步：对缺省数据进行处理；处理方式：
        方法一：删除特征
           1.如果特征值缺省率达到15%以上，则删除整列数据
           2.对其余缺省的特征列进行分析，删除关联较小的
        方法二：特征值补全
    第二步：对数据分析绘制的散点图异常数据的处理
    第三步：对每一列的特征进行分析，分别将其放入离散类别特征；连续类标特征和数据类型特征
    对于类别特征，将其类别和数量输出到控制台，人工进行判断
    对于数据类型和类别特征中的缺省值和异常值都进行处理
    :param data_list:
    """
    print(house_data_train.shape)
    na_count = house_data_train.isnull().sum().sort_values(ascending=False)
    na_rate = na_count / len(house_data_train)
    na_data = pd.concat([na_count, na_rate], axis=1, keys=['count', 'ratio'])
    # 方法一：需要删除的特征
    # df_train = train_data.drop(na_data[na_data['count'] > 1].index, axis=1)
    # print(df_train.shape)
    # 方法二
    house_data_train.drop(house_data_train[(house_data_train['OverallQual'] < 5) & (house_data_train['SalePrice'] > 200000)].index,
                    inplace=True)
    house_data_train.drop(house_data_train[(house_data_train['GrLivArea'] > 4000) & (house_data_train['SalePrice'] < 200000)].index,
                    inplace=True)
    house_data_train.drop(house_data_train[(house_data_train['YearBuilt'] < 1900) & (house_data_train['SalePrice'] > 400000)].index,
                    inplace=True)
    house_data_train.drop(house_data_train[(house_data_train['TotalBsmtSF'] > 6000) & (house_data_train['SalePrice'] < 200000)].index,
                    inplace=True)
    house_data_train.reset_index(drop=True, inplace=True)
    house_data_train.dropna(axis=1, how='any', inplace=True)
    # 第一步end
    contain_feature_list = []
    # 离散类别特征
    discrete_feature_list = []
    numeric_feature_list = []
    feature_map = {}
    for feature in house_data_train.columns[1:]:
        if house_data_train.dtypes[feature] == 'object':
            # 列的众数
            miss_feature_data = house_data_train[feature].mode()
            house_data_train.fillna(miss_feature_data, inplace=True)
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
        elif house_data_train.dtypes[feature] != 'object':
            # 使用列平均值进行填充
            miss_feature_data = house_data_train[feature].mean()
            house_data_train.fillna(miss_feature_data)
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
    # feature_handler()
    analysis_train_data()
    # feature_map = feature_handler(house_data_train)
    # feature_select(feature_map=feature_map, data_list=house_data_train)
