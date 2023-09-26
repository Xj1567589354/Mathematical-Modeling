import Train_Predict as trian_p
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

import pandas as pd
from sklearn import preprocessing

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


# 简单交叉验证, train_featrue可替换不同特征
def SimpleCrossValid(train_featrue = None, classifier = "LR", param=None):
    target = trian_p.get_target_df(train_featrue).copy()
    traindf = train_featrue.copy()
    # 切分数据 训练数据80% 验证数据20%
    train_all, test_all, train_target, test_target = trian_p.train_test_split(traindf, target,
                                                                              test_size=0.2, random_state=0)
    test_data = trian_p.get_predictors_df(test_all).copy()

    clf = trian_p.get_sklearn_model(classifier, param=param)
    test_pred = clf.predict_proba(test_data)[:, 1]

    score_test = roc_auc_score(test_target, test_pred)
    test_all['pred'] = test_pred

    print(classifier + "test 总体AUC:", score_test)
    print(classifier + "test Coupon AUC:", trian_p.myauc(test_all))


# K折交叉验证
def KFoldCrossValid(train_featrue = None, k = 5, classifier = "LR", param=None):
    clf = trian_p.get_sklearn_model(classifier, param=param)
    train = train_featrue.copy()
    target = trian_p.get_target_df(train_featrue).copy()
    kf = KFold(n_splits=k)

    scores = []
    score_coupons = []
    for k, (train_index, test_index) in enumerate(kf.split(train)):
        train_data, test_data, train_target, test_target = train.iloc[train_index], train.iloc[test_index], target[
            train_index], target[test_index]
        clf.fit(trian_p.get_predictors_df(train_data), train_target)
        test_pred = clf.predict_proba(trian_p.get_predictors_df(test_data))[:, 1]

        score_test = roc_auc_score(test_target, test_pred)
        test_data['pred'] = test_pred
        score_coupon_test = trian_p.myauc(test_data)

        scores.append(score_test)
        score_coupons.append(score_coupon_test)

    return scores, score_coupons


# 留P法交叉验证
def LPOCrossValid(train_featrue = None, p = 200, classifier = "LR", param=None):
    clf = trian_p.get_sklearn_model(classifier, param=param)
    train = train_featrue.copy()
    target = trian_p.get_target_df(train_featrue).copy()
    lpo = LeavePOut(p=p)

    scores = []
    score_coupons = []
    for k, (train_index, test_index) in enumerate(lpo.split(train)):
        train_data, test_data, train_target, test_target = train.iloc[train_index], train.iloc[test_index],\
                                                           target[train_index], target[test_index]
        clf.fit(trian_p.get_predictors_df(train_data), train_target)

        test_pred = clf.predict_proba(trian_p.get_predictors_df(test_data))[:, 1]
        score_test = roc_auc_score(test_target, test_pred)
        test_data['pred'] = test_pred
        score_coupon_test = trian_p.myauc(test_data)

        scores.append(score_test)
        score_coupons.append(score_coupon_test)
        if k >= 5:
            break

    return scores, score_coupons


# StratifieldKFlod交叉验证
def SKFoldCrossValid(train_featrue = None, k = 5, classifier = "LR", param=None):
    clf = trian_p.get_sklearn_model(classifier, param=param)
    train = train_featrue.copy()
    target = trian_p.get_target_df(train_featrue).copy()
    kf = StratifiedKFold(n_splits=k)

    scores = []
    score_coupons = []
    for k, (train_index, test_index) in enumerate(kf.split(train, target)):
        train_data, test_data, train_target, test_target = train.iloc[train_index], train.iloc[test_index], target[
            train_index], target[test_index]
        clf.fit(trian_p.get_predictors_df(train_data), train_target)

        test_pred = clf.predict_proba(trian_p.get_predictors_df(test_data))[:, 1]
        score_test = roc_auc_score(test_target, test_pred)
        test_data['pred'] = test_pred
        score_coupon_test = trian_p.myauc(test_data)

        scores.append(score_test)
        score_coupons.append(score_coupon_test)

    return scores, score_coupons


'''模型比较'''


# 对算法进行分析
def classifier_df_score(train_feat, classifier, cvnum, param=None):
    scores, score_coupons = SKFoldCrossValid(train_featrue=train_feat, classifier=classifier, k=cvnum, param=param)

    print(classifier+"总体AUC:", scores)
    print(classifier+"Coupon AUC:", score_coupons)


if __name__ == '__main__':
    # 数据读取
    train_f3 = pd.read_csv(r"F:\作业\研究生\数学建模\mathematical-modeling-group\models\data\3-b\train32.txt", sep='\t', encoding='utf-8')
    test_f3 = pd.read_csv(r"F:\作业\研究生\数学建模\mathematical-modeling-group\models\data\3-b\test32.txt", sep='\t', encoding='utf-8')
    train_f3, test_f3 = standize_df(train_f3, test_f3)

    classifier_df_score(train_feat=train_f1, classifier='LR', cvnum=5)                  # 计算AUC并打输出
    trian_p.plot_curve_single(train_f1, 'LR', 5, [0.01, 0.02, 0.05, 0.1, 0.2, 0.3])     # 绘制学习曲线
