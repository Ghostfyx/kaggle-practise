import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.linear_model import Lasso
from scipy.stats import skew

# pandas一些属性设置
pd.set_option('max_colwidth',200)
pd.set_option('display.width',200)
pd.set_option('display.max_columns',500)
pd.set_option('display.max_rows',1000)
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
    # 根据大神经验得出的四个关键属性
    draw_feature = ['OverallQual','GrLivArea', 'YearBuilt','TotalBsmtSF']
    # 绘制散点图
    # sns.pairplot(x_vars=top_feature_list, y_vars=['SalePrice'], data=house_data_train)
    # enumerate同时遍历索引和内容
    for index, feature in enumerate(draw_feature):
        plt.figure(figsize=(23, 13))
        plt.scatter(x=house_data_train[feature], y=house_data_train["SalePrice"])
        # figure作用：新建画布
        plt.figure(figsize=(18.5, 10.5))
        sns.boxplot(x=house_data_train[feature], y=house_data_train["SalePrice"])
        plt.show()
    plt.show()

# data_visualization()
"""
下面开始进行数据清洗：
第一步：通过对散点图数据的观察，删除异常点数据
"""
house_data_train.drop(house_data_train[(house_data_train['OverallQual']<5) & (house_data_train['SalePrice']>200000)].index,inplace=True)
house_data_train.drop(house_data_train[(house_data_train['GrLivArea']>4000) & (house_data_train['SalePrice']<200000)].index,inplace=True)
house_data_train.drop(house_data_train[(house_data_train['YearBuilt']<1900) & (house_data_train['SalePrice']>400000)].index,inplace=True)
house_data_train.drop(house_data_train[(house_data_train['TotalBsmtSF']>6000) & (house_data_train['SalePrice']<200000)].index,inplace=True)
house_data_train.reset_index(drop=True, inplace=True)
train_target = np.log(house_data_train['SalePrice'])
"""
第二步：对缺省数据进行处理；处理方式：
    方法一：删除特征
        1.如果特征值缺省率达到15%以上，则删除整列数据
        2.对其余缺省的特征列进行分析，删除关联较小的
    方法二：特征值补全
        1.对于类型变量根据类别说明进行填充
        2.对于连续型变量，采取平均值，众数等处理
        3.对于离散型变量
"""
train_size = house_data_train.shape[0]
house_data = pd.concat([house_data_train, house_data_test], ignore_index=True)
house_data.drop(['Id'],axis=1, inplace=True)
print(house_data.columns)
house_data_count = house_data.isnull().sum().sort_values(ascending=False)
train_data_count = house_data_train.isnull().sum().sort_values(ascending=False)
house_data_rate = house_data_count / len(house_data)
train_data_rate = train_data_count / len(house_data_train)
house_na_data = pd.concat([house_data_count, house_data_rate], axis=1, keys=['count', 'ratio'])
train_na_data = pd.concat([train_data_count, train_data_rate], axis=1, keys=['count', 'ratio'])
# print(house_na_data[house_na_data['count']>0])
# print(train_na_data)
full_feature_list = list(house_na_data[(house_na_data['ratio'] <= 0.5) & (house_na_data['ratio'] > 0)].index)
drop_feature_list = list(house_na_data[house_na_data['ratio'] > 0.5].index)
print(drop_feature_list)
# 方法一：需要删除的特征
drop_feature_list = ['PoolQC', 'MiscFeature', 'Alley', 'Fence']
house_data.drop(drop_feature_list, axis=1, inplace=True)
print(house_data.shape)
"""
方法二：特征填充，将类别特征变量和数字特征变量区分开来，首先针对不同的缺失特征进行填充
用所有相同邻居的住宅的距离中位数来填充
"""
house_data['LotFrontage'] = house_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
house_data['LotFrontage'] = house_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
# Electrical只有一个缺失值所有用众数填充
house_data['Electrical'] = house_data['Electrical'].fillna(value=house_data['Electrical'].mode()[0])
#对于车库缺省值即为没有车库
Garage_feature = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'GarageYrBlt']
house_data[Garage_feature] = house_data[Garage_feature].fillna(value="without")
# 对于地下室缺省值即为没有地下室
Bsmt_feature = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
house_data[Bsmt_feature] = house_data[Bsmt_feature].fillna(value="without")
# 对于壁炉缺省即为没有壁炉
house_data['FireplaceQu'] = house_data['FireplaceQu'].fillna(value='without')
# 对于地板类型和面积，对地板类型用众数填充，对地板面积使用对应地面类型面积的中位数填充
house_data['MasVnrType'] = house_data['MasVnrType'].fillna(value='None')
house_data['MasVnrArea'] = house_data['MasVnrArea'].fillna(value=0)
house_data['MSZoning'] = house_data['MSZoning'].fillna(value=house_data['MSZoning'].mode()[0])
house_data['BsmtHalfBath'] = house_data['BsmtHalfBath'].fillna(value=house_data['BsmtHalfBath'].mode()[0])
house_data['Utilities'] = house_data['Utilities'].fillna(value=house_data['Utilities'].mode()[0])
house_data['Functional'] = house_data['Functional'].fillna(value=house_data['Functional'].mode()[0])
house_data['BsmtFullBath'] = house_data['BsmtFullBath'].fillna(value=house_data['BsmtFullBath'].mode()[0])
house_data['BsmtFinSF2'] = house_data['BsmtFinSF2'].fillna(value=0)
house_data['BsmtFinSF1'] = house_data['BsmtFinSF1'].fillna(value=0)
house_data['Exterior2nd'] = house_data['Exterior2nd'].fillna(value=house_data_train['Exterior2nd'].mode()[0])
house_data['BsmtUnfSF'] = house_data['BsmtUnfSF'].fillna(value=0)
house_data['TotalBsmtSF'] = house_data['TotalBsmtSF'].fillna(value=0)
house_data['Exterior1st'] = house_data['Exterior1st'].fillna(value=house_data_train['Exterior1st'].mode()[0])
house_data['SaleType'] = house_data['SaleType'].fillna(value=house_data_train['SaleType'].mode()[0])
house_data['KitchenQual'] = house_data['KitchenQual'].fillna(value=house_data_train['KitchenQual'].mode()[0])
house_data['GarageArea'] = house_data['GarageArea'].fillna(value=0)
house_data['GarageCars'] = house_data['GarageCars'].fillna(value=0)
# print(house_data.isnull().sum().sort_values(ascending=False))

"""
开始特征工程
"""
print("开始特征工程")
# 对于具有程度特征的类别特征按照从好到坏的顺序进行转换

NumStr = ["MSSubClass","BsmtFullBath","BsmtHalfBath","HalfBath","BedroomAbvGr","KitchenAbvGr","MoSold","YrSold","YearBuilt","YearRemodAdd","LowQualFinSF","GarageYrBlt"]
house_data[NumStr] = house_data[NumStr].astype(str)
house_data = house_data.replace({'Utilities': {'AllPub': 4, 'NoSeWa': 3, 'NoSewr': 2, 'ELO': 1},
                             'Street': {'Pave': 2, 'Grvl': 1},
                             'FireplaceQu': {'Ex': 6,
                                            'Gd': 5,
                                            'TA': 4,
                                            'Fa': 3,
                                            'Po': 1,
                                            'without': 1
                                            },
                             'Fence': {'GdPrv': 4,
                                       'GdWo': 3,
                                       'MnPrv': 2,
                                       'MnWw': 2,
                                       'NoFence': 1},
                             'ExterQual': {'Ex': 5,
                                            'Gd': 4,
                                            'TA': 3,
                                            'Fa': 2,
                                            'Po': 1
                                            },
                             'ExterCond': {'Ex': 5,
                                            'Gd': 4,
                                            'TA': 3,
                                            'Fa': 2,
                                            'Po': 1
                                            },
                             'BsmtQual': {'Ex': 6,
                                            'Gd': 5,
                                            'TA': 4,
                                            'Fa': 3,
                                            'Po': 2,
                                            'without': 1},
                             'BsmtExposure': {'Gd': 4,
                                            'Av': 3,
                                            'Mn': 2,
                                            'No': 1,
                                            'without': 1},
                             'BsmtCond': {'Ex': 6,
                                            'Gd': 5,
                                            'TA': 4,
                                            'Fa': 3,
                                            'Po': 2,
                                            'without': 1},
                             'GarageQual': {'Ex': 6,
                                            'Gd': 5,
                                            'TA': 4,
                                            'Fa': 3,
                                            'Po': 2,
                                            'without': 1},
                             'GarageCond': {'Ex': 6,
                                            'Gd': 5,
                                            'TA': 4,
                                            'Fa': 3,
                                            'Po': 2,
                                            'without': 1},
                             'KitchenQual': {'Ex': 5,
                                            'Gd': 4,
                                            'TA': 3,
                                            'Fa': 2,
                                            'Po': 1},
                             'Functional': {'Typ': 1,
                                            'Min1': 2,
                                            'Min2': 2,
                                            'Mod': 3,
                                            'Maj1': 4,
                                            'Maj2': 5,
                                            'Sev': 6,
                                            'Sal': 7},
                             'GarageType' :{
                                            'CarPort':1,
                                            'None':1,
                                           'Detchd':2,
                                           '2Types':3,
                                            'Basment':3,
                                           'Attchd':4,
                                            'BuiltIn':5},
                             'GarageFinish':{
                                            'None':1,
                                             'Unf':2,
                                             'RFn':3,
                                             'Fin':4 },
                            'PavedDrive':{'N':1, 'P':2, 'Y':3},
                            'SaleCondition':{'AdjLand':1, 'Abnorml':2, 'Alloca':2, 'Family':2, 'Normal':3, 'Partial':4},
                            'MasVnrType':{'BrkCmn':1, 'None':1, 'BrkFace':2, 'Stone':3},
                            'Exterior1st':{'BrkComm':1,
                                             'AsphShn':2, 'CBlock':2, 'AsbShng':2,
                                             'WdShing':3, 'Wd Sdng':3, 'MetalSd':3, 'Stucco':3, 'HdBoard':3,
                                             'BrkFace':4, 'Plywood':4,
                                             'VinylSd':5,
                                             'CemntBd':6,
                                             'Stone':7, 'ImStucc':7},
                            'MSSubClass':{'180':1,
                                        '30':2, '45':2,
                                        '190':3, '50':3, '90':3,
                                        '85':4, '40':4, '160':4,
                                        '70':5, '20':5, '75':5, '80':5, '150':5,
                                        '120': 6, '60':6}
                        })
house_data["SaleType"] = house_data.SaleType.map({'COD': 1, 'ConLD': 1, 'ConLI': 1, 'ConLw': 1, 'Oth': 1, 'WD': 1,
                                       'CWD': 2, 'Con': 3, 'New': 3})
# 添加部分特征
house_data['TotalSF'] = house_data['TotalBsmtSF'] + house_data['1stFlrSF'] + house_data['2ndFlrSF']
house_data['YearsSinceRemodel'] = house_data['YrSold'].astype(int) - house_data['YearRemodAdd'].astype(int)
house_data['Total_Home_Quality'] = house_data['OverallQual'] + house_data['OverallCond']
house_data2 = house_data.copy()
lab = LabelEncoder()
house_data2["YearBuilt"] = lab.fit_transform(house_data2["YearBuilt"])
house_data2["YearRemodAdd"] = lab.fit_transform(house_data2["YearRemodAdd"])
house_data2["GarageYrBlt"] = lab.fit_transform(house_data2["GarageYrBlt"])
# 计算各个特征的偏移量（偏移量的作用见笔记）
house_data2_numeric = house_data2.select_dtypes(exclude=["object"])
skewness = house_data2_numeric.apply(lambda x: skew(x))
skewness_features = skewness[abs(skewness) >= 1].index
house_data2[skewness_features] = np.log1p(house_data2[skewness_features])
# 对离散型没有连续关系的类别特征进行one-hot编码，会根据类别特征的个数增加新的一列
house_data2 = pd.get_dummies(house_data2)
print(house_data2.head())
house_data = house_data2
del house_data2
# 使用来对数据进行标准化处理，RobustScaler比standard更加准确
scaler = RobustScaler()
train = house_data.iloc[:train_size]
print(train.shape)
print(train_target.shape)
test = house_data[:train_size]
train_scaled = scaler.fit(train).transform(train)
test_scaled = scaler.transform(test)
print("数据预处理完毕")
print("开始特征降维")
# pca = PCA(n_components =20, copy=True)
# data_list_pca = pca.fit(train_scaled)
# print(pca.explained_variance_ratio_)
# print(pca.explained_variance_)
lasso=Lasso(alpha=0.001)
lasso.fit(train_scaled,train_target)
feature_importance = pd.DataFrame({"Feature Importance":lasso.coef_}, index=train.columns).sort_values("Feature Importance",ascending=False)
print(feature_importance)
