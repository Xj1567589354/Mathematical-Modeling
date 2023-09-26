from sklearn import preprocessing
import pandas as pd
from sklearn import metrics
from lightgbm import LGBMClassifier
import numpy as np
from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.svm import SVC


id_col_names = ['user_id', 'coupon_id', 'date_received']                # id列
target_col_name = 'label'                                               # target--label列
id_target_cols = ['user_id', 'coupon_id', 'date_received', 'label']     # target--四列id
myeval = 'roc_auc'
cvscore=0
# 目录定义
datapath = '../data/'               # 数据路径
featurepath = datapath + 'feature/'         # 特征路径
resultpath = '../result/'           # 结果路径
tmppath = '../tmp/'
scorepath = '../score/'             # 预测得分路径


# 返回ID列
def get_id_df(df):
    return df[id_col_names]


# 返回Target列
def get_target_df(df):
    return df[target_col_name]


# 返回特征列
def get_predictors_df(df):
    predictors = [f for f in df.columns if f not in id_target_cols]
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


# 特征归一化
def standize_df(train_data, test_data):
    features_columns = [f for f in test_data.columns if f not in id_target_cols]
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler = min_max_scaler.fit(train_data[features_columns])

    train_data_scaler = min_max_scaler.transform(train_data[features_columns])
    test_data_scaler = min_max_scaler.transform(test_data[features_columns])

    train_data_scaler = pd.DataFrame(train_data_scaler)
    train_data_scaler.columns = features_columns

    test_data_scaler = pd.DataFrame(test_data_scaler)
    test_data_scaler.columns = features_columns

    train_data_scaler['label'] = train_data['label']
    train_data_scaler[id_col_names] = train_data[id_col_names]
    test_data_scaler[id_col_names] = test_data[id_col_names]
    return train_data_scaler, test_data_scaler

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


# roc曲线
def roc(y_label, y_pre):

    fpr, tpr, thersholds = roc_curve(y_label, y_pre, pos_label=2)

    for i, value in enumerate(thersholds):
        print("%f %f %f" % (fpr[i], tpr[i], value))

    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)

    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


train_f3, test_f3 = read_data('sf3')
train_f3, test_f3 = standize_df(train_f3, test_f3)

# 按日期划分数据集
train = train_f3[train_f3.date_received < 20160515].copy()
test = train_f3[train_f3.date_received >= 20160515].copy()

# 获取数据
train_data = get_predictors_df(train).copy()
train_target = get_target_df(train).copy()
test_data = get_predictors_df(test).copy()
test_target = get_target_df(test).copy()

clf = LGBMClassifier(max_depth=5, min_child_samples=10, reg_alpha=0.1)  # 构建模型
clf.fit(train_data, train_target)  # 训练

# 测试集上进行预测
result = clf.predict_proba(test_data)[:, 1]
predict = clf.predict(test_data)

test['pred'] = result
score = metrics.roc_auc_score(test_target, result)  # 获取总体AUC值
score_coupon = myauc(test)                          # 平均AUC值
print("LGBMClassifier 总体 AUC:", score)
print("LGBMClassifier Coupon AUC:", score_coupon)
print("accuracy_score:", accuracy_score(test_target, predict))
print("f1_score:", metrics.f1_score(test_target, predict))

# 测试集上进行预测
result = clf.predict_proba(test_f3)[:, 1]
predict = clf.predict(test_f3)
print("s")
