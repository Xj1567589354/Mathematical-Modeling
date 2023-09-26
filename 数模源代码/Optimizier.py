"""模型调参"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split    # 切分数据
import Train_Predict as trian_p
from sklearn import metrics
from sklearn.model_selection import learning_curve

# SKLearn 集成的算法
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

import pandas as pd
from sklearn import preprocessing

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# 返回Target列
def get_target_df(df):
    return df['target']


# 返回特征列
def get_predictors_df(df):
    predictors = [f for f in df.columns if f not in ['target']]
    return df[predictors]


# 特征归一化
def standize_df(train_data, test_data):
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
    return train_data_scaler, test_data_scaler


def custom_accuracy_score(y_true, y_pred, decimals=2):
    correct = 0
    total = len(y_true)

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1

    accuracy = correct / total
    rounded_accuracy = round(accuracy, decimals)

    return rounded_accuracy


# 构建模型
def get_sklearn_model(model_name, params=None):
    if model_name == 'NB':            # 朴素贝叶斯
        model = MultinomialNB(alpha=0.01)
    elif model_name == 'LR':          # 逻辑
        model = LogisticRegression(penalty='l2')
    elif model_name == 'KNN':         # KNN
        model = KNeighborsClassifier(n_neighbors=5, weights='distance')
    elif model_name == 'RF':          # 随机森林
        model = RandomForestClassifier(n_estimators=120, max_depth=9, min_samples_split=5, min_samples_leaf=1)
    elif model_name == 'DT':          # 决策树
        model = tree.DecisionTreeClassifier()
    elif model_name == 'SVC':         # 向量机
        model = SVC()
    elif model_name == 'GBDT':        # GBDT
        model = GradientBoostingClassifier()
    elif model_name == 'XGB':         # XGBoost
        model = XGBClassifier(objective= 'multi:softmax', num_class= 7, max_depth=3, min_child_weight=1,
                              n_estimators=100)
    elif model_name == 'LGB':         # LGBoost
        model = LGBMClassifier(verbosity = -1, objective= 'multiclass', num_class= 7, boosting_type='gbdt',
                               max_depth=2, n_estimators=250, num_leaves=20, learning_rate=0.1)
    elif model_name == 'MLP':
        model = MLPClassifier(hidden_layer_sizes=(100, 200))
    else:
        print("wrong model name!")
        return
    if params is not None:
        model.set_params(**params)
    return model


# 穷举网格搜索
def GridSearch(classifier, params, train_data, train_target, val_data, val_target, test_data):

    """这里采用的是随机森林，可替换成其他模型，比如LGB"""
    model = get_sklearn_model(model_name=classifier)
    parameters = params
    stratified_cv = StratifiedKFold(n_splits=5)

    grid_search = GridSearchCV(model, parameters, cv=stratified_cv, verbose=2)
    grid_search.fit(train_data, train_target)

    predict = grid_search.predict(val_data)

    test_predicted = grid_search.predict(test_data)

    # 获取最优模型和参数
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    plot_learning_curve(classifier=best_model, data=train_f3)        # 绘制学习曲线

    # 输出最优模型和参数
    print("最优参数:")
    print("Best Model:", best_model)
    print("Best Parameters:", best_params)
    sorted(grid_search.cv_results_.keys())

    # score_test = metrics.roc_auc_score(val_target, predict)
    scores = cross_val_score(best_model, val_data, val_target, cv=stratified_cv)

    # print("RandomForestClassifier GridSearchCV test AUC:   ", score_test)
    print("Cross-validation scores:", scores)
    accuracy = custom_accuracy_score(y_true=val_target, y_pred=predict)
    precision = precision_score(y_true=val_target, y_pred=predict, average='macro')
    recall = recall_score(y_true=val_target, y_pred=predict, average='macro')
    f1_score = metrics.f1_score(val_target, predict, average='macro')
    print("Val accuracy_score:", accuracy)
    print("Val precision_score:", precision)
    print("Val recall_score:", recall)
    print("Val f1_score: {:.8f}".format(f1_score))

    return test_predicted


# # 随机搜索
# def RandomSearch(params=None):
#     model = RandomForestClassifier()
#     '''
#     当上一次参数搜索出最优之后，把参数加入到模型当中，继续针对不同的参数进一步的进行搜索
#     往复执行，直到所有参数都搜索到最优为止
#     '''
#     parameters = params
#
#     clf = RandomizedSearchCV(model, parameters, cv=3, verbose=2)
#     clf.fit(train_data, train_target)
#
#     score_test = metrics.roc_auc_score(test_target, clf.predict(test_data))
#
#     print("RandomForestClassifier RandomizedSearchCV test AUC:   ", score_test)
#     print("最优参数:")
#     print(clf.best_params_)
#     sorted(clf.cv_results_.keys())


"""验证曲线--可视化调参过程"""


# 验证曲线
def grid_plot(train_feat, classifier, cvnum, param_range, param_name, param=None):
    from sklearn.model_selection import validation_curve
    train_scores, test_scores = validation_curve(
        trian_p.get_sklearn_model(classifier, param), trian_p.get_predictors_df(train_feat),
        trian_p.get_target_df(train_feat),
        param_name=param_name, param_range=param_range,
        cv=cvnum, scoring='roc_auc', n_jobs=1)
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


# 学习曲线
def plot_learning_curve(classifier, data):
    estimator = classifier
    train_data = get_predictors_df(data)
    train_target = get_target_df(data)
    stratified_cv = StratifiedKFold(n_splits=5)
    train_sizes, train_scores, test_scores = learning_curve(estimator, train_data, train_target, cv=stratified_cv, scoring='accuracy')

    # 计算性能指标的均值和标准差
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # 绘制学习曲线
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training Accuracy')
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='Validation Accuracy')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')

    # 设置图表标题和坐标轴标签
    plt.title('Learning Curve')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('KNN' + '学习曲线2.png', dpi=500)
    plt.show()


if __name__ == '__main__':
    # 数据读取
    train_f3 = pd.read_csv(r"F:\作业\研究生\数学建模\mathematical-modeling-group\models\data\3-b\train32.txt", sep='\t', encoding='utf-8')
    test_f3 = pd.read_csv(r"F:\作业\研究生\数学建模\mathematical-modeling-group\models\data\3-b\test32.txt", sep='\t', encoding='utf-8')
    train_f3, test_f3 = standize_df(train_f3, test_f3)

    target = get_target_df(train_f3).copy()

    # 随机划分数据集，20%概率
    train_all, val_all, train_target, val_target = train_test_split(train_f3, target, test_size=0.2, random_state=0)

    train_data = get_predictors_df(train_all).copy()
    val_data = get_predictors_df(val_all).copy()
    test_data = get_predictors_df(test_f3).copy()

    # 绘制调参验证曲线
    params = {

    }

    """两种搜索二选一"""
    # 采用网格搜索
    predicts = GridSearch(classifier='KNN', params=params, train_data=train_data, train_target=train_target, val_data=val_data,
               val_target=val_target, test_data=test_data)

    print(predicts)

    # # 采用随机搜索
    # RandomSearch(params={'n_estimators': [20, 50, 100], 'max_depth': [1, 2, 3]})


