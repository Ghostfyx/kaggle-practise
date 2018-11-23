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

# 与价格相关性较高的属性
file_path = '../../data/houseprice/train.csv'
house_data_train = pd.read_csv(file_path)
house_corr = house_data_train.corr()

def ananalysis_train_data():
    """
    数据分析
    :rtype: object
    """
    print("train data length----->" + str(len(house_data_train)))
    feature_list = house_data_train.columns
    print(feature_list)
    print(house_data_train)
    # 默认n=5,n表示查看的行数
    print(house_data_train.head(n=5))
    # 对训练数据的房子价格进行分析，计算平均值，标准化值等
    print(house_data_train["SalePrice"].describe())
    # 得到与SalePrice相关性top10的特征属性
    corr_feature_sort = house_corr['SalePrice'].sort_values(ascending=False)
    top_feature_list = list(corr_feature_sort[(corr_feature_sort.values > 0.5) & (corr_feature_sort.values < 1)].index)
    # 'OverallQual,装修与房子质量', 'GrLivArea,住房面积', 'GarageCars,车库的车容量', 'GarageArea,车库大小', 'TotalBsmtSF,地下室面积',
    # '1stFlrSF,一楼面积', 'FullBath，全身的浴室数量', 'TotRmsAbvGrd,房间总数（不含浴室）', 'YearBuilt,建筑年代', 'YearRemodAdd,重构日期'
    return top_feature_list

def data_visualization(top_feature_list):
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
    feature_list = house_data_train.columns
    print("Skewness: %f" % house_data_train['SalePrice'].skew())
    print("Kurtosis: %f" % house_data_train['SalePrice'].kurt())
    # 绘制相关系数矩阵
    plt.figure(figsize=(18.5, 10.5))
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
    for index, feature in enumerate(feature_list):
        plt.subplot(10, 8, index+1)
        sns.stripplot(x=feature, y="SalePrice", data=house_data_train)
    # figure作用：新建画布
    plt.figure(figsize=(18.5, 10.5))
    # 绘制上述变量与价格的箱型图
    for index, feature in enumerate(feature_list):
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
            1.对于类型变量采用，“MISSING”填充
            2.对于连续型变量，采取平均值，众数等处理
            3.对于离散型变量
    第二步：对数据分析绘制的散点图异常数据的处理
    对于数据类型和类别特征中的缺省值和异常值都进行处理
    """
    print(house_data_train.shape)
    na_count = house_data_train.isnull().sum().sort_values(ascending=False)
    na_rate = na_count / len(house_data_train)
    na_data = pd.concat([na_count, na_rate], axis=1, keys=['count', 'ratio'])
    miss_feature_list = list(na_data[na_data['count'] > 0].index)
    # print(miss_feature_list)
    # 方法一：需要删除的特征
    # df_train = train_data.drop(na_data[na_data['count'] > 1].index, axis=1)
    # print(df_train.shape)
    # 方法二：特征填充
    for feature in miss_feature_list:
        if house_data_train.dtypes[feature] == 'object':
            house_data_train[feature] = house_data_train[feature].astype('category')
            house_data_train[feature] = house_data_train[feature].cat.add_categories('MISSING')
            """
            对于dataform：0 or 'index', 1 or 'columns'；对于Series不需要
            not_object_feature = house_data_train.isnull().sum().sort_values(ascending=False)
            print(not_object_feature[not_object_feature > 3])
            剩余三个非对像类型的属性：LotFrontage,GarageYrBlt,MasVnrArea(砖石面积)
            """
            house_data_train[feature].fillna(value='MISSING', inplace=True)
        else:
            mean = int(house_data_train[feature].mean())
            house_data_train[feature].fillna(value=mean, inplace=True)
    print(house_data_train.isnull().sum().sort_values(ascending=False))


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
    # top_feature_list = ananalysis_train_data()
    # data_visualization(top_feature_list)
    feature_handler()
