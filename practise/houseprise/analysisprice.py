import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
import warnings

from sklearn.svm import SVR, LinearSVR
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge, ElasticNet, RidgeCV, LassoCV, ElasticNetCV, \
    BayesianRidge, SGDRegressor
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
"""
数据分析
:rtype: object
"""
print("train data length----->" + str(len(house_data_train)))
# 默认n=5,n表示查看的行数
# print(house_data_train.head())
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
print(house_data.shape)
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
house_data["MSZoning"] = house_data.MSZoning.map({'C (all)':1, 'RH':2, 'RM':2, 'RL':3, 'FV':4})
house_data["Neighborhood"] = house_data.Neighborhood.map({'MeadowV': 1,
                                               'IDOTRR': 2, 'BrDale': 2,
                                               'OldTown': 3, 'Edwards': 3, 'BrkSide': 3,
                                               'Sawyer': 4, 'Blueste': 4, 'SWISU': 4, 'NAmes': 4,
                                               'NPkVill': 5, 'Mitchel': 5,
                                               'SawyerW': 6, 'Gilbert': 6, 'NWAmes': 6,
                                               'Blmngtn': 7, 'CollgCr': 7, 'ClearCr': 7, 'Crawfor': 7,
                                               'Veenker': 8, 'Somerst': 8, 'Timber': 8,
                                               'StoneBr': 9,
                                               'NoRidge': 10, 'NridgHt': 10})
house_data["Condition1"] = house_data.Condition1.map({'Artery': 1,
                                           'Feedr': 2, 'RRAe': 2,
                                           'Norm': 3, 'RRAn': 3,
                                           'PosN': 4, 'RRNe': 4,
                                           'PosA': 5, 'RRNn': 5})
# 添加部分特征
# house_data['TotalSF'] = house_data['TotalBsmtSF'] + house_data['1stFlrSF'] + house_data['2ndFlrSF']
# house_data['YearsSinceRemodel'] = house_data['YrSold'].astype(int) - house_data['YearRemodAdd'].astype(int)
# house_data['Total_Home_Quality'] = house_data['OverallQual'] + house_data['OverallCond']
house_data.drop(columns = ['SalePrice'], inplace = True)
house_data2 = house_data.copy()
"""
scikit-learn扩展通常要继承BaseEstimator，下面我们创建自己的转换器《Python数据挖掘入门与实战》会有实例
以后的kaggle练习中我会多使用这样的转换器，
"""
class labelencode(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        lab = LabelEncoder()
        X["YearBuilt"] = lab.fit_transform(X["YearBuilt"])
        X["YearRemodAdd"] = lab.fit_transform(X["YearRemodAdd"])
        X["GarageYrBlt"] = lab.fit_transform(X["GarageYrBlt"])
        return X

class skew_dummies(BaseEstimator, TransformerMixin):
    def __init__(self, skew=0.5):
        self.skew = skew

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_numeric = X.select_dtypes(exclude=["object"])
        skewness = X_numeric.apply(lambda x: skew(x))
        skewness_features = skewness[abs(skewness) >= self.skew].index
        X[skewness_features] = np.log1p(X[skewness_features])
        X = pd.get_dummies(X)
        return X

# 我们对一些特征进行组合，得到新的特征
class add_feature(BaseEstimator, TransformerMixin):
    def __init__(self, additional=1):
        self.additional = additional

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.additional == 1:
            X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
            X["TotalArea"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]

        else:
            X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
            X["TotalArea"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]

            X["+_TotalHouse_OverallQual"] = X["TotalHouse"] * X["OverallQual"]
            X["+_GrLivArea_OverallQual"] = X["GrLivArea"] * X["OverallQual"]
            X["+_MSZoning_TotalHouse"] = X["MSZoning"] * X["TotalHouse"]
            X["+_MSZoning_OverallQual"] = X["MSZoning"] + X["OverallQual"]
            X["+_MSZoning_YearBuilt"] = X["MSZoning"] + X["YearBuilt"]
            X["+_Neighborhood_TotalHouse"] = X["Neighborhood"] * X["TotalHouse"]
            X["+_Neighborhood_OverallQual"] = X["Neighborhood"] + X["OverallQual"]
            X["+_Neighborhood_YearBuilt"] = X["Neighborhood"] + X["YearBuilt"]
            X["+_BsmtFinSF1_OverallQual"] = X["BsmtFinSF1"] * X["OverallQual"]
            X["-_Functional_TotalHouse"] = X["Functional"] * X["TotalHouse"]
            X["-_Functional_OverallQual"] = X["Functional"] + X["OverallQual"]
            X["-_LotArea_OverallQual"] = X["LotArea"] * X["OverallQual"]
            X["-_TotalHouse_LotArea"] = X["TotalHouse"] + X["LotArea"]
            X["-_Condition1_TotalHouse"] = X["Condition1"] * X["TotalHouse"]
            X["-_Condition1_OverallQual"] = X["Condition1"] + X["OverallQual"]
            X["Bsmt"] = X["BsmtFinSF1"] + X["BsmtFinSF2"] + X["BsmtUnfSF"]
            X["Rooms"] = X["FullBath"] + X["TotRmsAbvGrd"]
            X["PorchArea"] = X["OpenPorchSF"] + X["EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"]
            X["TotalPlace"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"] + X["OpenPorchSF"] + X[
                "EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"]
            print(X.shape)
            return X
pipe = Pipeline([
    ('labelencode', labelencode()),
    ('add_feature', add_feature(additional=2)),
    ('skew_dummies', skew_dummies(skew=1)),
    ])
house_data2 = pipe.fit_transform(house_data2)
# 对离散型没有连续关系的类别特征进行one-hot编码，会根据类别特征的个数增加新的一列
house_data = house_data2
del house_data2
print(house_data.shape)
# 使用来对数据进行标准化处理，RobustScaler比standard更加准确
scaler = RobustScaler()
train = house_data[:train_size]
test = house_data[:train_size]
train_scaled = scaler.fit(train).transform(train)
test_scaled = scaler.transform(test)
print("数据预处理完毕")
print("开始特征降维")
# print("使用Lasso回归通过特征的参数向0收缩的程度判断特征的重要性")
# lasso = Lasso(alpha=0.001)
# lasso.fit(train_scaled,train_target)
# feature_importance = pd.DataFrame({"Feature Importance" : lasso.coef_}, index=train.columns).sort_values("Feature Importance",ascending=False)
# feature_importance[feature_importance["Feature Importance"]!=0].plot(kind="barh",figsize=(15,30))
# plt.xticks(rotation=90)
# plt.show()
print("训练集特征维数:%d" %train.shape[1])
# 使用PCA降维，维数的设置采用交叉验证的方式，二分法交叉验证
# 100-150得分提升，150-200扥分下降
step = [50,100,150,200]
setp_2 = [150,160,170,180,190]
step_3 = [150,151,152]
for item in step_3:
    pca = PCA(n_components=item)
    score = cross_val_score(pca, train_scaled,train_target, cv=5).mean()
    print('降维数为：%d的平均值为：%d'%(item, score))
pca = PCA(n_components=150)
train_pca = pca.fit_transform(train_scaled)
test_pca = pca.transform(test_scaled)
# 对模型选取进行交叉验证，打算选取模型：Lasso回归，rige回归，ElasticNet，LinearRegression，XgBoost
print("使用交叉验证获取模型参数：")
alphas_step1 = [0.01,0.03,0.09,0.1,0.3,0.9,1]
alphas_step2 = [1,2,3,4,5,6,7,8]
alphas_step3 = [75,76,77,78,79,80,81,82,83,84,85]
alphas_step4 = [0.003,0.004,0.005,0.055,0.06]
liner_r = LinearRegression()
# 选取ridge模型的正则化超参数
ridge_cv1 = RidgeCV(alphas=alphas_step1, cv=5)
ridge_cv1.fit(train_pca, train_target)
print("ridge model1 parameter alpha: %.4f"%(ridge_cv1.alpha_))
ridge_cv2 = RidgeCV(alphas=alphas_step2, cv=5)
ridge_cv2.fit(train_pca, train_target)
print("ridge model2 parameter alpha: %.4f"%(ridge_cv2.alpha_))
ridge_cv = RidgeCV(alphas=alphas_step3, cv=5)
ridge_cv.fit(train_pca, train_target)
print("ridge model parameter alpha: %.4f"%(ridge_cv.alpha_))
lasso_cv = LassoCV(alphas=alphas_step4,max_iter=10000)
gbr = GradientBoostingRegressor()
ela_cv = ElasticNetCV(alphas=alphas_step4,max_iter=10000)
ela_cv.fit(train_pca, train_target)
print("Elacv model parameter alpha: %.4f"%(ela_cv.alpha_))
# 默认高斯核函数
svr = SVR()
xgb = XGBRegressor()
names = ["LR", "Ridge", "Lasso","GBR","Ela",'svr',"xgb"]
models = [liner_r, ridge_cv, lasso_cv, gbr,ela_cv, svr, xgb]
print(train_pca.shape)
print(train_target.shape)
def rmse_cv(model,X,y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    return rmse
print("降维后：")
for name, model in zip(names, models):
    model.fit(train_pca,train_target)
    # 均方损失函数：neg_mean_squared_error
    score = rmse_cv(model, train_pca, train_target)
    print("{}: {:.4f}, {:.4f}".format(name, score.mean(), score.std()))
print("降维前：")
for name, model in zip(names, models):
    score = rmse_cv(model, train_scaled, train_target)
    print("{}: {:.6f}, {:.4f}".format(name,score.mean(),score.std()))
# 总的来说树模型普遍不如线性模型，可能还是因为get_dummies（独热编码one-hot编码）后带来的数据稀疏性，不过这些模型都是没调过参的。
# 下面对第一步筛选出的模型进行调参
print("使用网络搜索选取模型假设的超参数:")
class grid():
    def __init__(self, model):
        self.model = model

    def grid_get(self, X, y, param_grid):
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring="neg_mean_squared_error")
        grid_search.fit(X, y)
        print('best model and parameter:')
        print(grid_search.best_estimator_,grid_search.best_params_, np.sqrt(-grid_search.best_score_))
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        result_pd = pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score', 'std_test_score']]
        print(result_pd)
print('model Lasso:')
grid(Lasso()).grid_get(train_scaled,train_target,{'alpha': [0.0004,0.0005,0.0007,0.0006,0.0009,0.0008],'max_iter':[5000,7500,10000,12500,15000,17500,20000]})
print('model Ridge:')
grid(Ridge()).grid_get(train_scaled,train_target,{'alpha':[35,40,45,50,55,60,65,70,80,90]})
print('model SVR:')
grid(SVR()).grid_get(train_scaled,train_target,{'C':[11,12,13,14,15],'kernel':["rbf"],"gamma":[0.0003,0.0004],"epsilon":[0.008,0.009]})
print('model KernelRidge:')
param_grid={'alpha':[0.2,0.3,0.4,0.5], 'kernel':["polynomial"], 'degree':[3],'coef0':[0.8,1,1.2]}
grid(KernelRidge()).grid_get(train_scaled,train_target,param_grid)
print('model ElasticNet:')
grid(ElasticNet()).grid_get(train_scaled,train_target,{'alpha':[0.0005,0.0008,0.004,0.005],'l1_ratio':[0.08,0.1,0.3,0.5,0.7],'max_iter':[5000,7500,10000,12500,15000,17500,20000]})
print("对已经经过调参的各个模型进行分析比较，选取模型及其参数：")
lasso = Lasso(alpha=0.0009,max_iter=5000)
ridge = Ridge(alpha=80)
svr = SVR(gamma= 0.0004,kernel='rbf',C=14,epsilon=0.008)
ker = KernelRidge(alpha=0.5 ,kernel='polynomial',degree=3 , coef0=1.2)
ela = ElasticNet(alpha=0.004,l1_ratio=0.3,max_iter=5000)
bay = BayesianRidge()
names = ["LR", "Ridge", "Lasso", "SVR", "Ela","Bay","Ker"]
models = [LinearRegression(), ridge, lasso, svr, ela, bay, ker]
print('使用全网搜索选择参数后，对模型进行评价：')
for name, model in zip(names, models):
    score = rmse_cv(model, train_scaled, train_target)
    print("{}: {:.6f}, {:.4f}".format(name,score.mean(),score.std()))
"""
模型构建已经完成了模型对训练数据的拟合和模型超参数的调节，到现在为止进行了：1.数据集分析，重点分析需要预测的特征；2.数据可视化；
3.数据清洗，包括：异常点的剔除，缺省值的填充，类别变量特征的转换（连续型和离散型），数值类别特征的归一化处理；
4.分析特征集，关联部分特征；组成新的特征；此时注意新增特征特征都是和原始特征高度相关的，这可能导致较强的多重共线性 (Multicollinearity)
5.数据集是否需要降维处理；PCA也可以去除特征的共线性！！
6.模型的选择：从简单模型出发，从模型对训练集的拟合能力和对测试集的泛化能力出发。模型选择的衡量参数：
7.模型超参数的选择：可以使用交叉验证和全网搜索的方法
8.模型的组合，即为集成学习方法，
"""
print('使用集成学习方法进行最后处理：')
# 需要对集成模型进行调参，现在直接使用别人的参数，后续会学习xgboost调参
# print("Boosting方法的xgBoost模型")
# model_xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
#                              learning_rate=0.05, max_depth=3,
#                              min_child_weight=1.7817, n_estimators=2200,
#                              reg_alpha=0.4640, reg_lambda=0.8571,
#                              subsample=0.5213, silent=1,
#                               nthread = -1)
# score = rmse_cv(model_xgb, train_scaled, train_target)
# print("{}: {:.6f}, {:.4f}".format(name,score.mean(),score.std()))
print('使用Bagging方法的randomforest回归模型模型：')
# 首先对randomforestregressor调参，使用网络搜索的方法
param_grid = {'max_features':[.1,.3,.5,.7,.9,.99],'n_estimators':[100,200,300,400,500,600,700],
              'oob_score':[True,False]}
grid(RandomForestRegressor(n_jobs=-1, random_state=50)).grid_get(train_scaled,train_target,param_grid)
# 使用stacking方法集成不同的学习模型进行学习
print('stacking方法开始：')