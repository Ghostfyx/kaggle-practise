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
train_file_path = '../../data/houseprice/train.csv'
test_file_path = '../../data/houseprice/test.csv'
house_data_train = pd.read_csv(train_file_path)
house_data_test = pd.read_csv(test_file_path)
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


def data_visualization():
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
    feature_list = house_data_train.columns[1:]
    print("Skewness: %f" % house_data_train['SalePrice'].skew())
    print("Kurtosis: %f" % house_data_train['SalePrice'].kurt())
    # 绘制相关系数矩阵
    plt.figure(figsize=(18.5, 10.5))
    # vmin, vmax : 显示的数据值的最大和最小的范围
    sns.heatmap(house_corr, vmax=1, square=True)
    # 绘制相关系数矩阵
    plt.figure(figsize=(18.5, 10.5))
    k = len(feature_list)
    # nlargest的优点就是能一次看到最大的几行
    cols = house_corr.nlargest(k, columns='SalePrice')['SalePrice'].index
    cm = np.corrcoef(house_data_train[cols].values.T)
    sns.set(font_scale=1.25)
    sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                yticklabels=cols.values, xticklabels=cols.values)
    draw_feature = ['OverallQual','GrLivArea', 'YearBuilt','TotalBsmtSF']
    # 绘制散点图
    # sns.pairplot(x_vars=top_feature_list, y_vars=['SalePrice'], data=house_data_train)
    # enumerate同时遍历索引和内容
    for index, feature in enumerate(draw_feature):
        plt.figure(figsize=(23, 13))
        sns.stripplot(x=feature, y="SalePrice", data=house_data_train)
        plt.show()
    # figure作用：新建画布
    plt.figure(figsize=(18.5, 10.5))
    # 绘制上述变量与价格的箱型图
    for index, feature in enumerate(draw_feature):
        data = pd.concat([house_data_train['SalePrice'], house_data_train[feature]], axis=1)
        plt.subplot(2, 2, index+1)
        sns.boxplot(x=feature, y="SalePrice", data=data)
    plt.show()

# data_visualization()

"""
通过对散点图数据的观察，删除异常点数据
"""
house_data_train.drop(house_data_train[(house_data_train['OverallQual']<5) & (house_data_train['SalePrice']>200000)].index,inplace=True)
house_data_train.drop(house_data_train[(house_data_train['GrLivArea']>4000) & (house_data_train['SalePrice']<200000)].index,inplace=True)
house_data_train.drop(house_data_train[(house_data_train['YearBuilt']<1900) & (house_data_train['SalePrice']>400000)].index,inplace=True)
house_data_train.drop(house_data_train[(house_data_train['TotalBsmtSF']>6000) & (house_data_train['SalePrice']<200000)].index,inplace=True)
house_data_train.reset_index(drop=True, inplace=True)
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
train_size = house_data_train.shape[0]
house_data = pd.concat([house_data_train, house_data_test]).reset_index(drop=True)
print(house_data.shape)
house_data = house_data.isnull().sum().sort_values(ascending=False)
house_data_rate = house_data / len(house_data)
house_data_count = house_data.isnull().sum().sort_values(ascending=False)
house_na_data = pd.concat([house_data_count, house_data_rate], axis=1, keys=['count', 'ratio'])
print(house_na_data)
full_feature_list = list(house_na_data[(house_na_data['ratio'] <= 0.5) & (house_na_data['ratio'] > 0)].index)
drop_feature_list = list(house_na_data[house_na_data['ratio'] > 0.5].index)
print(drop_feature_list)
# 方法一：需要删除的特征
drop_feature_list.append("Id")
house_data.drop(drop_feature_list, axis=1, inplace=True)
print(house_data.shape)
"""
方法二：特征填充，将类别特征变量和数字特征变量区分开来，首先针对不同的缺失特征进行填充
用所有相同邻居的住宅的距离中位数来填充
"""
house_data_train['LotFrontage'] = house_data_train.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
house_data_test['LotFrontage'] = house_data_test.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
# Electrical只有一个缺失值所有用众数填充
house_data_train['Electrical'] = house_data_train['Electrical'].fillna(value=house_data_train['Electrical'].mode()[0], inplace=True)
#对于车库缺省值即为没有车库
Garage_feature = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
house_data_train[Garage_feature] = house_data_train[Garage_feature].fillna(value="None", inplace=True)
house_data_test[Garage_feature] = house_data_test[Garage_feature].fillna(value="None", inplace=True)
# 对于地板类型和面积，对地板类型用众数填充，对地板面积使用对应地面类型面积的中位数填充
print(house_data_train['MasVnrType'])
print(house_data_train['MasVnrType'].mode())
house_data_train['MasVnrType'] = house_data_train['MasVnrType'].fillna(value=house_data_train['MasVnrType'].mode()[0], inplace=True)
house_data_train['MasVnrArea'] = house_data_train.groupby('MasVnrType')['MasVnrArea'].transform(lambda x: x.fillna(x.median()))
# 对于地下室缺省值即为没有地下室
Bsmt_feature = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
house_data_train[Bsmt_feature] = house_data_train[Bsmt_feature].fillna(value="None", inplace=True)
house_data_test[Bsmt_feature] = house_data_test[Bsmt_feature].fillna(value="None", inplace=True)
# 对于壁炉缺省即为没有壁炉
house_data_train['FireplaceQu'] = house_data_train['FireplaceQu'].fillna(value='None', inplace=True)
house_data_test['FireplaceQu'] = house_data_test['FireplaceQu'].fillna(value='None', inplace=True)
# for feature in quality:
#     house_data_train[feature] = house_data_train[feature].astype('category')
#     house_data_train[feature] = house_data_train[feature].cat.add_categories('MISSING')
#     """
#     对于dataform：0 or 'index', 1 or 'columns'；对于Series不需要
#     not_object_feature = house_data_train.isnull().sum().sort_values(ascending=False)
#     print(not_object_feature[not_object_feature > 3])
#     剩余三个非对像类型的属性：LotFrontage,GarageYrBlt,MasVnrArea(砖石面积)
#     """
#     house_data_train[feature].fillna(value='MISSING', inplace=True)
# house_data_train[quantity] = house_data_train[quantity].fillna(0.)
test_na_feature = ['MSZoning','Utilities','KitchenQual','Functional','SaleType','Exterior1st', 'Exterior2nd']
house_data_test[test_na_feature] = house_data_test[test_na_feature].fillna(value='None', inplace=True)
print(house_data_train.isnull().sum().sort_values(ascending=False))
print(house_data_test.isnull().sum().sort_values(ascending=False))

"""
对所有类型变量，依照各个类型变量的不同取值对应的样本集内房价的均值，
按照房价均值高低对此变量的当前取值确定其相对数值1,2,3,4等。
相当于对类型变量赋值使其成为连续变量。
此方法采用了与One-Hot编码不同的方法来处理离散数据，
:param quality_feature:
"""


"""
第一步，对训练集进行数据预处理，找出数据类型的特征和非数据类型的特征
:param  data_list:训练数据集
:param feature_map:特征类型集合
"""
# standard = StandardScaler()
# vec = LabelEncoder()
# for feature in feature_map['contain_feature']:
#     a = data_list[feature].values
#     data_list[feature] = vec.fit_transform(data_list[feature].values)
# for feature in feature_map['discrete_feature']:
#     a = data_list[feature].values
#     data_list[feature] = vec.fit_transform(data_list[feature].values)
# for feature in feature_map['numeric_feature']:
#     # 注意此处reshape的作用：将数据转换为1列
#     a = data_list[feature].values
#     data_list[feature] = standard.fit_transform(data_list[feature].values.reshape(-1, 1))
# print("数据预处理完毕")
# pca = PCA(n_components =10, copy=True)
# data = data_list.iloc[:, :-1]
# data_list_pca = pca.fit(data)
# print(pca.explained_variance_ratio_)
# print(pca.explained_variance_)

