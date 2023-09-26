import pandas as pd
import warnings
from sklearn import preprocessing

from sklearn.linear_model import LinearRegression       # 线性回归
from sklearn.neighbors import KNeighborsRegressor       # K近邻回归
from sklearn.tree import DecisionTreeRegressor          # 决策树回归
from sklearn.ensemble import RandomForestRegressor      # 随机森林回归
from sklearn.svm import SVR                             # 支持向量回归
import lightgbm as lgb                                  # lightGbm模型
from sklearn.linear_model import SGDRegressor

from sklearn.model_selection import train_test_split    # 切分数据
from sklearn.metrics import mean_squared_error          # 评价指标

from sklearn.decomposition import PCA                   # 主成分分析法


# 构建模型
def get_skelearn_model(model_name, param=None, n=None):
    if model_name == 'LR':                                  # 多元线性回归
        model = LinearRegression()
    elif model_name == 'KNN':                               # KNN
        model = KNeighborsRegressor(n_neighbors=n)
    elif model_name == 'DT':                                # 决策树
        model = DecisionTreeRegressor()
    elif model_name == 'RF':                                # 随机森林
        model = RandomForestRegressor(n_estimators=n)
    elif model_name == 'LGB':                               # LGBoost
        model = lgb.LGBMRegressor()
    elif model_name == 'SGD':
        # SGD，这里使用了l2正则化，消除欠拟合过拟合的影响；其他正则化处理还有：l1, elasticnet
        model = SGDRegressor(max_iter=1000, tol=1e-3, penalty= 'l2')
    elif model_name == 'SVR':
        model = SVR()
    else:
        print("wrong model name!")
        return
    if param is not None:
        model.set_params(**param)
    return model


def train_model(regressor=None, n=None, params=None):
    # 构建模型并训练
    if regressor == 'LGB':
        clf = get_skelearn_model(model_name=regressor, param=params)
        clf.fit(X=train_data, y=train_target, eval_metric='MSE')
    else:
        clf = get_skelearn_model(model_name=regressor, n=n, param=params)
        clf.fit(X=train_data, y=train_target)

    # 输出结果
    score = mean_squared_error(test_target, clf.predict(test_data))
    print(regressor + ":", score)


if __name__ == '__main__':
    """
    导入数据，该部分承接数据预处理和特征工程步骤
    数据处理+切分数据
    """
    warnings.filterwarnings("ignore")
    # 读取数据路径
    train_data_file = r"F:\作业\研究生\数学建模\mathematical-modeling-group\models\data\train.txt"
    test_data_file = r"F:\作业\研究生\数学建模\mathematical-modeling-group\models\data\test.txt"

    # 读取数据
    train_data = pd.read_csv(train_data_file, sep='\t', encoding='utf-8')
    test_data = pd.read_csv(test_data_file, sep='\t', encoding='utf-8')
    features_columns = [col for col in train_data.columns if col not in ['target']]

    # 正态分布校验
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler = min_max_scaler.fit(train_data[features_columns])
    train_data_scaler = min_max_scaler.transform(train_data[features_columns])
    test_data_scaler = min_max_scaler.transform(test_data[features_columns])
    train_data_scaler = pd.DataFrame(train_data_scaler)
    train_data_scaler.columns = features_columns
    test_data_scaler = pd.DataFrame(test_data_scaler)
    test_data_scaler.columns = features_columns
    train_data_scaler['target'] = train_data['target']

    # PCA方法降维
    # 保留16个主成分
    pca = PCA(n_components=16)
    new_train_pca_16 = pca.fit_transform(train_data_scaler.iloc[:, 0:-1])
    new_test_pca_16 = pca.transform(test_data_scaler)
    new_train_pca_16 = pd.DataFrame(new_train_pca_16)
    new_test_pca_16 = pd.DataFrame(new_test_pca_16)
    new_train_pca_16['target'] = train_data_scaler['target']

    # 采用 pca 保留16维特征的数据
    new_train_pca_16 = new_train_pca_16.fillna(0)
    train = new_train_pca_16[new_test_pca_16.columns]
    target = new_train_pca_16['target']

    # 切分数据 训练数据80% 验证数据20%
    train_data, test_data, train_target, test_target = train_test_split(train, target, test_size=0.2, random_state=0)

    # 这里举例的是LGB
    train_model(regressor='LGB', params={
            'learning_rate': 0.0001,
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'max_depth': -1,
            'n_estimators': 1000,
            'random_state': 2019
        })

