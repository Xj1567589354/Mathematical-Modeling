import pandas as pd
import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot as plt
import Train as T

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split        # 切分数据
from sklearn.metrics import mean_squared_error              # 评价指标
from sklearn.model_selection import validation_curve

from sklearn.decomposition import PCA   # 主成分分析法
import lightgbm as lgb


# 穷举网格搜索
def GridSearch(params=None):
    model = lgb.LGBMRegressor(num_leaves=31)
    parameters = params
    # 切分数据 训练数据80% 验证数据20%
    train_data, test_data, train_target, test_target = train_test_split(train, target, test_size=0.2, random_state=0)

    grid_search = GridSearchCV(model, parameters, cv=5)
    grid_search.fit(train_data, train_target)

    print("Test set score:{:.2f}".format(grid_search.score(test_data, test_target)))
    print("Best parameters:{}".format(grid_search.best_params_))
    print("Best score on train set:{:.2f}".format(grid_search.best_score_))
    sorted(grid_search.cv_results_.keys())


# 随机搜索
def RandomSearch(params=None):
    model = lgb.LGBMRegressor(num_leaves=31)
    parameters = params
    # 切分数据 训练数据80% 验证数据20%
    train_data, test_data, train_target, test_target = train_test_split(train, target, test_size=0.2, random_state=0)

    grid_search = RandomizedSearchCV(model, parameters, cv=5)
    grid_search.fit(train_data, train_target)

    print("Test set score:{:.2f}".format(grid_search.score(test_data, test_target)))
    print("Best parameters:{}".format(grid_search.best_params_))
    print("Best score on train set:{:.2f}".format(grid_search.best_score_))
    sorted(grid_search.cv_results_.keys())


def grid_plot(data, target, classifier, cvnum, param_range, param_name, param=None):
    train_scores, test_scores = validation_curve(
        T.get_skelearn_model(model_name=classifier, param=param), data, target, param_name=param_name,
        param_range=param_range, cv=cvnum, scoring='r2', n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with " + classifier)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    plt.semilogx(param_range, train_scores_mean, label="Training score", color="r")
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="r")
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="g")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.legend(loc="best")
    plt.show()


if __name__ == '__main__':
    # 读取数据
    train_data_file = r"F:\作业\研究生\数学建模\mathematical-modeling-group\models\data\train.txt"
    test_data_file = r"F:\作业\研究生\数学建模\mathematical-modeling-group\models\data\test.txt"
    train_data = pd.read_csv(train_data_file, sep='\t', encoding='utf-8')
    test_data = pd.read_csv(test_data_file, sep='\t', encoding='utf-8')

    # 正态分布校验
    features_columns = [col for col in train_data.columns if col not in ['target']]
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

    # 搜索参数
    params = {
        'learning_rate': [0.0001, 0.001, 0.01],
        'n_estimators': [40, 200, 100, 20]
    }

    # 网格搜索
    RandomSearch(params=params)

    """这里列举的是LGB的验证曲线"""

    X = train_data.values
    y = train_target.values
    params = {
        'learning_rate': 0.01,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'sub_feature': 0.6,
        'num_leaves': 50,
        'bagging_fraction': 0.8,
        'feature_fraction': 0.8
    }

    grid_plot(data=X, target=y, classifier='LGB', cvnum=10,
              param_name='n_estimators', param_range=[50, 100, 150, 200])





