import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer

def analysis_train_data():
    file_path = '../../data/houseprice/train.csv'
    house_data_train = pd.read_csv(file_path)
    print("train data length----->"+str(len(house_data_train)))
    feature_list = house_data_train.columns
    print(feature_list)
    # 默认n=5,n表示查看的行数
    print(house_data_train.head(n=5))
    # 对训练数据的房子价格进行分析，计算平均值，标准化值等
    print(house_data_train["SalePrice"].describe())
    # 利用数据可视化绘制房子价格的直方图，根据直方图分析房价的数据分布
    # 发现房价数据正偏，偏离正太分布，有峰值
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
    # plt.show()
    '''
    分析各个特征对房价关联程度，找出影响房价较大的几个特征，本次打算使用PCA主成分析
    利用协方差矩阵
    '''
    return house_data_train


def feature_handler(data_list):
    """
    对每一列的特征进行分析，分别将其放入离散类别特征；连续类标特征和数据类型特征
    对于类别特征，将其类别和数量输出到控制台，人工进行判断
    对于数据类型和类别特征中的缺省值和异常值都进行处理
    :param data_list:
    """
    # 连续类标特征
    contain_feature_list = []
    # 离散类别特征
    discrete_feature_list = []
    numeric_feature_list = []
    feature_map = {}
    for feature in data_list.columns[1:]:
        if data_list.dtypes[feature] == 'object':
            # value_counts是pandas自带的统计值函数
            feature_value_counts = data_list[feature].value_counts()
            print(feature+'包含的类别个数为 ', len(feature_value_counts))
            print(feature_value_counts.index)
            label = input("请输入英文字母c或者d，c表虚连续类别，d表示离散类别：")
            if label == 'c':
                contain_feature_list.appdend(feature)
            elif label == 'd':
                discrete_feature_list.append(feature)
            else:
                break
        elif data_list.dtypes[feature] != 'object':
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
    vec = DictVectorizer()
    for feature in feature_map['contain_feature']:
        data_list_format = vec.fit_transform(data_list[feature].values.reshape(-1, 1))
        data_list[feature].values = data_list_format
    for feature in feature_map['discrete_feature']:
        a = data_list[feature].values.reshape(-1, 1)
        data_list_format = vec.fit_transform(a)
    for feature in feature_map['numeric_feature']:
        # 注意此处reshape的作用：将数据转换为1列
        data_list_format = standard.fit_transform(data_list[feature].values.reshape(-1, 1))
    pca = PCA(n_components=10)
    data_list_pca = pca.fit_transform(data_list)
    print(data_list_pca)
    return data_list_pca



if __name__ == '__main__':
    house_data_train = analysis_train_data()
    feature_map = feature_handler(house_data_train)
    feature_select(feature_map=feature_map, data_list=house_data_train)
