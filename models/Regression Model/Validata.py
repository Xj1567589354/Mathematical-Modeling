import pandas as pd
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
import numpy as np
import Train as T

from sklearn.model_selection import train_test_split    # 切分数据
from sklearn.metrics import mean_squared_error          # 评价指标
from sklearn.decomposition import PCA   # 主成分分析法
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import KFold               # 5折交叉验证
from sklearn.model_selection import LeavePOut

"""使用了交叉验证和L1L2加权正则化处理欠拟合和过拟合问题"""


# 简单交叉验证, train_featrue可替换不同特征
def SimpleCrossValid(classifier = "SGD", param=None):
    clf = T.get_skelearn_model(model_name=classifier,
                               param=param)
    clf.fit(train_data, train_target)
    score_train = mean_squared_error(train_target, clf.predict(train_data))
    score_test = mean_squared_error(test_target, clf.predict(test_data))
    print(classifier + " train MSE:   ", score_train)
    print(classifier + " test MSE:   ", score_test)


# K折交叉验证
def KFoldCrossValid(k = 5, classifier = "LR", param=None):
    kf = KFold(n_splits=k)
    for k, (train_index, test_index) in enumerate(kf.split(train)):
        train_data, test_data, train_target, test_target = train.values[train_index], train.values[test_index], target[
            train_index], target[test_index]
        clf = T.get_skelearn_model(model_name=classifier,
                                   param=param)
        clf.fit(train_data, train_target)
        score_train = mean_squared_error(train_target, clf.predict(train_data))
        score_test = mean_squared_error(test_target, clf.predict(test_data))
        print(k, " 折", classifier + " train MSE:   ", score_train)
        print(k, " 折", classifier + " test MSE:   ", score_test, '\n')


# 留P法交叉验证
def LPOCrossValid(p = 10, classifier = "LR", param=None):
    lpo = LeavePOut(p=p)
    for k, (train_index, test_index) in enumerate(lpo.split(train)):
        train_data, test_data, train_target, test_target = train.values[train_index], train.values[test_index], target[
            train_index], target[test_index]
        clf = T.get_skelearn_model(model_name=classifier,
                                   param=param)
        clf.fit(train_data, train_target)
        score_train = mean_squared_error(train_target, clf.predict(train_data))
        score_test = mean_squared_error(test_target, clf.predict(test_data))
        print(k, " 10个", classifier + " train MSE:   ", score_train)
        print(k, " 10个", classifier + " test MSE:   ", score_test, '\n')
        if k >= 9:
            break


# 学习曲线
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    print(train_scores_mean)
    print(test_scores_mean)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    return plt


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

    # # 这里举例的留P法交叉验证
    # LPOCrossValid(classifier='SGD')

    """这里列举的是LR的学习曲线"""
    X = train_data.values
    y = train_target.values

    title = r"LR"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    estimator = T.get_skelearn_model(model_name='LR')  # 建模
    plot_learning_curve(estimator, title, X, y, ylim=(0, 1), cv=cv, n_jobs=1)
