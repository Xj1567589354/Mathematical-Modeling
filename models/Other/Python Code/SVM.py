import pandas as pd
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn import svm
from sklearn.model_selection import GridSearchCV
# from lift_curve import lift_curve
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
import random
import math
from sklearn import preprocessing
from sklearn.decomposition import PCA                   # 主成分分析法


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
X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.2, random_state=0)


# # 不平衡问题处理
# # SMOTE法过采样
# smote = SMOTE(random_state=0)  # random_state为0（此数字没有特殊含义，可以换成其他数字）使得每次代码运行的结果保持一致
# X_smo, y_smo = smote.fit_resample(X_train, y_train)  # 使用原始数据的特征变量和目标变量生成过采样数据集
# print('SMOTE过采样后标签数据统计：', Counter(y_smo))
#
# # 随机过采样
# ros = RandomOverSampler(random_state=0, sampling_strategy='auto')
# X_ros, y_ros = ros.fit_resample(X_train, y_train)
# print('随机过采样后标签数据统计：', Counter(y_ros))


param_grid = {
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'C': [1],
    'gamma': [1],
}

# 支持向量机
model_svm = svm.SVC(probability=True)
svm_cv = GridSearchCV(estimator=model_svm, param_grid=param_grid,
                      scoring='roc_auc', cv=4)
print('SVM最优参数', svm_cv.best_params_)


# 使用SVM对测试集进行预测
# 得到测试集每个样本为正例的概率（在这里用的是predict_proba，其他模型中也可以用decision_function，
# 要看GridSearchCV套用的模型的类中有哪些方法）
y_score = svm_cv.predict_proba(X_test)[:, 1]
y_pre = svm_cv.predict(X_test)
print('SVM精确度...')
print(metrics.classification_report(y_test, y_pre))
print('SVM AUC...')
fpr, tpr, th = metrics.roc_curve(y_test, y_score)  # 构造 roc 曲线，第三个输出为阈值（每个阈值对应一个DPR和TPR）
ks = max(tpr - fpr)
print('KS=', ks)
print('AUC = %.4f' %metrics.auc(fpr, tpr))  # 求AUC值(ROC曲线下方面积)

# 画ROC曲线
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()

# # 画lift曲线
# lift_curve(y_test, y_score, 10)
