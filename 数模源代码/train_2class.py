from sklearn import metrics
import numpy as np
import pandas as pd
# import datetime
# from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
# import lightgbm as lgb
# import xgboost as xgb
import matplotlib.pyplot as plt
# from sklearn.naive_bayes import GaussianNB
# from sklearn.model_selection import learning_curve
# from sklearn.model_selection import ShuffleSplit

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

from sklearn.model_selection import train_test_split    # 切分数据
# from sklearn.metrics import mean_squared_error          # 评价指标
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.model_selection import learning_curve
# from sklearn.model_selection import ShuffleSplit
import warnings
warnings.filterwarnings("ignore")

'''
变量定义
'''
# 全局参数
id_col_names = ['user_id', 'coupon_id', 'date_received']                # id列
target_col_name = 'target'                                               # target--label列
id_target_cols = ['user_id', 'coupon_id', 'date_received', 'label']     # target--四列id
myeval = 'roc_auc'
cvscore=0
# 目录定义
datapath = '../data/'               # 数据路径
featurepath = datapath + 'feature/'         # 特征路径
resultpath = '../result/'           # 结果路径
tmppath = '../tmp/'
scorepath = '../score/'             # 预测得分路径


'''特征读取'''


# 返回ID列
def get_id_df(df):
    return df[id_col_names]


# 返回Target列
def get_target_df(df):
    return df[target_col_name]


# 返回特征列
def get_predictors_df(df):
    predictors = [f for f in df.columns if f not in ['target']]
    return df[predictors]


# 按特征名读取训练集
def read_featurefile_train(featurename):
    df=pd.read_csv(featurepath+'train_'+featurename+'.csv', sep=',' , encoding = "utf-8")
    df.fillna(0, inplace=True)
    return df


# 按特征名读取测试集
def read_featurefile_test(featurename):
    df=pd.read_csv(featurepath+'test_'+featurename+'.csv', sep=',' , encoding = "utf-8")
    df.fillna(0, inplace=True)
    return df


# 按特征名读取数据
def read_data(featurename):
    traindf = read_featurefile_train(featurename)
    testdf = read_featurefile_test(featurename)
    return traindf, testdf


'''模型训练'''


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


# 构建模型
def get_sklearn_model(model_name, param=None):
    if model_name == 'NB':            # 朴素贝叶斯
        model = MultinomialNB(alpha=0.01)
    elif model_name == 'LR':          # 逻辑
        model = LogisticRegression(penalty='l2')
    elif model_name == 'KNN':         # KNN
        model = KNeighborsClassifier()
    elif model_name == 'RF':          # 随机森林
        model = RandomForestClassifier()
    elif model_name == 'DT':          # 决策树
        model = tree.DecisionTreeClassifier()
    elif model_name == 'SVC':         # 向量机
        model = SVC()
    elif model_name == 'GBDT':        # GBDT
        model = GradientBoostingClassifier()
    elif model_name == 'XGB':         # XGBoost
        model = XGBClassifier()
    elif model_name == 'LGB':         # LGBoost
        model = LGBMClassifier()
    elif model_name == 'MLP':
        model = MLPClassifier()
    else:
        print("wrong model name!")
        return
    if param is not None:
        model.set_params(**param)
    return model


# 画学习曲线
def plot_learning_curve(estimator, title, x, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, x, y, cv=cv, scoring=myeval, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
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


# 画算法的学习曲线,为加快画图速度，最多选20%数据
def plot_curve_single(traindf, classifier, cvnum, train_sizes = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]):
    X = get_predictors_df(traindf)
    y = get_target_df(traindf)
    estimator = get_sklearn_model(classifier)    # 建模
    title = "learning curve of "+classifier+", cv:"+str(cvnum)
    plot_learning_curve(estimator, title, X, y, ylim=(0, 1.01), cv=cvnum, train_sizes=train_sizes)


# 性能评价函数
# 本赛题目标是预测投放的优惠券是否核销。
# 针对此任务及一些相关背景知识，使用优惠券核销预测的平均AUC（ROC曲线下面积）作为评价标准。
# 即对每个优惠券coupon_id单独计算核销预测的AUC值，再对所有优惠券的AUC值求平均作为最终的评价标准。
# coupon平均auc计算
def myauc(test):
    testgroup = test.groupby(['coupon_id'])
    aucs = []
    for i in testgroup:
        coupon_df = i[1]
        # 测算AUC必须大于1个类别
        if len(coupon_df['label'].unique()) < 2:
            continue
        auc = metrics.roc_auc_score(coupon_df['label'], coupon_df['pred'])
        aucs.append(auc)
    return np.average(aucs)


def custom_accuracy_score(y_true, y_pred, decimals=2):
    correct = 0
    total = len(y_true)

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1

    accuracy = correct / total
    rounded_accuracy = round(accuracy, decimals)

    return rounded_accuracy


def train_model_split(traindf, classifier, params=None):
    target = get_target_df(traindf).copy()

    # 随机划分数据集，20%概率
    train_all, val_all, train_target, val_target = train_test_split(traindf, target, test_size=0.2, random_state=0)

    train_data = get_predictors_df(train_all).copy()
    val_data = get_predictors_df(val_all).copy()

    clf = get_sklearn_model(classifier, param=params)
    clf.fit(train_data, train_target)
    predict = clf.predict(val_data)
    result = clf.predict_proba(val_data)[:, 1]

    score = metrics.roc_auc_score(val_target, result,)
    accuracy = custom_accuracy_score(y_true=val_target, y_pred=predict)
    precision = precision_score(y_true=val_target, y_pred=predict)
    recall = recall_score(y_true=val_target, y_pred=predict)
    f1_score = metrics.f1_score(val_target, predict)
    print(classifier + "总体AUC: ", score)
    print("Val accuracy_score:", accuracy)
    print("Val precision_score:", precision)
    print("Val recall_score:", recall)
    print("Val f1_score: {:.8f}".format(f1_score))

    return accuracy, precision, score, f1_score, recall


'''模型验证，结果输出'''


# 预测函数
def classifier_df_simple(train_feat, test_feat, classifier, params=None):
    model = get_sklearn_model(classifier, param=params)
    model.fit(get_predictors_df(train_feat), get_target_df(train_feat))
    predicted = pd.DataFrame(model.predict_proba(get_predictors_df(test_feat))[:, 1])
    return round(predicted, 4)


# 输出结果
def output_metric_log(accuracy, auc, f1_score, classifier):
    # 创建包含指标和预测值的数据框
    df = pd.DataFrame({'Model': [classifier], 'Accuracy': [accuracy], 'AUC': [auc], 'F1': [f1_score]})
    # 读取已有的CSV文件（如果存在）
    try:
        existing_data = pd.read_csv(r'F:\作业\研究生\数学建模\mathematical-modeling-group\models\Classification Model\metrics_log_redu.csv')
        df = pd.concat([existing_data, df], ignore_index=True)  # 将新数据和已有数据合并
    except FileNotFoundError:
        pass

    # 将数据框保存为CSV文件
    df.to_csv('metrics_log_redu.csv', index=False)


# def output_prediciton_log(predicts, classifier):
#     # 创建包含指标和预测值的数据框
#     list=[]
#     list.append('Model')
#     for i in range(1, 161):
#         list.append('Prediction'+str(i))
#
#     # 将数据框保存为CSV文件
#     df = pd.DataFrame(columns=list)
#     # print(predicts.iloc[:,0].tolist())
#     predicts = predicts.iloc[:, 0].tolist()
#     predicts.insert(0, classifier)
#
#     df.loc[len(df)] = predicts
#
#     df.to_csv('prediction_log.csv.py', index=False)


if __name__ == '__main__':
    '''
    以下特征都是在数据预处理和特征工程处生成的
    用不同特征进行训练，对比效果
    '''
    # train_f1, test_f1 = read_data('f1')                     # 读取数据
    # train_f1, test_f1 = standize_df(train_f1, test_f1)      # 归一化数据，以便KNN进行预测
    #
    # train_f2, test_f2 = read_data('sf2')
    # train_f2, test_f2 = standize_df(train_f2, test_f2)

    train_f3 = pd.read_csv(r"F:\作业\研究生\数学建模\mathematical-modeling-group\models\data\1-b\train2.txt", sep='\t', encoding='utf-8')
    test_f3 = pd.read_csv(r"F:\作业\研究生\数学建模\mathematical-modeling-group\models\data\1-b\test2.txt", sep='\t', encoding='utf-8')
    train_f3, test_f3 = standize_df(train_f3, test_f3)

    '''使用不同模型进行训练预测，对比效果'''
    # LGB
    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'verbosity': -1,  # 设置verbosity为-1，以隐藏警告信息
        'learning_rate': 0.1,
        'max_depth': 10,
        'num_class': 7,
    }

    # # LGB
    # params = {
    #     'num_leaves': 60,
    #     'min_data_in_leaf': 30,
    #     'objective': 'multiclass',
    #     'num_class': 7,
    #     'max_depth': 10,
    #     'learning_rate': 0.03,
    #     "min_sum_hessian_in_leaf": 6,
    #     "boosting": "gbdt",
    #     "feature_fraction": 0.9,
    #     "bagging_freq": 1,
    #     "bagging_fraction": 0.8,
    #     "bagging_seed": 11,
    #     "lambda_l1": 0.1,
    #     "verbosity": -1,
    #     "nthread": 15,
    #     'metric': 'multi_logloss',
    #     "random_state": 2019,
    # }

    # # XGB
    # params = {
    #     'objective': 'binary:logistic',
    #     'eval_metric': 'logloss',
    #     'max_depth': 3,
    #     'learning_rate': 0.1,
    #     'subsample': 0.8,
    #     'colsample_bytree': 0.8,
    #     'gamma': 0,
    #     'reg_alpha': 0,
    #     'reg_lambda': 1
    # }

    # # RF
    # params = {
    #     'n_estimators': 100,
    #     'max_depth': None,
    #     'min_samples_split': 2,
    #     'min_samples_leaf': 1,
    #     'max_features': 'auto',
    #     'bootstrap': True,
    #     'random_state': 42
    # }

    # # SVC
    # params = {
    #     'C': 1.0,
    #     'kernel': 'rbf',
    #     'gamma': 'scale',
    #     'degree': 3,
    #     'probability': True,
    #     'class_weight': None
    # }

    # # MLP
    # params = {
    #     'hidden_layer_sizes': (200, 100),
    #     'activation': 'relu',
    #     'solver': 'adam',
    #     'alpha': 0.0001,
    #     'learning_rate': 'constant',
    #     'max_iter': 200,
    #     'random_state': 42,
    #     'tol': 1e-4,
    #     'verbose': False,
    #     'early_stopping': False
    # }

    accuracy, auc, f1_score, precision, recall = train_model_split(train_f3, 'LGB', params)     # 训练，train_f3可替换成前文三种特征中的任意一种，LGB模型也可替换其他模型

    # '''预测结果'''
    predicted = classifier_df_simple(train_f3, test_f3, 'LGB', params)
    predicted = predicted.transpose()
    print(predicted)
    # output_metric_log(accuracy, auc, f1_score, classifier='SVC')
    # output_prediciton_log(predicts=predicted, classifier='XGB')
    # plot_curve_single(train_f3, 'LGB', 5, [0.01, 0.02, 0.05, 0.1, 0.2, 0.3])        # 绘制学习曲线


