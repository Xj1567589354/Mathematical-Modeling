import json
import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split    # 切分数据
from sklearn.decomposition import PCA                   # 主成分分析法
from sklearn.svm import SVR


# MAPE
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


# SMAPE
def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100


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


train_f3 = pd.read_csv(r"F:\作业\研究生\数学建模\mathematical-modeling-group\models\data\1-b\train2.txt", sep='\t', encoding='utf-8')
test_f3 = pd.read_csv(r"F:\作业\研究生\数学建模\mathematical-modeling-group\models\data\1-b\test2.txt", sep='\t', encoding='utf-8')
train_f3, test_f3 = standize_df(train_f3, test_f3)

# 切分数据 训练数据80% 验证数据20%
train_data, test_data, train_target, test_target = train_test_split(train_f3, test_f3, test_size=0.2, random_state=0)

model = SVR()

params={'kernel': 'rbf',
        'C': 1.0,
        'epsilon': 0.2}

model.set_params(**params)
model.fit(X=train_data, y=train_target)
y_predict = model.predict(X=test_data)
print("MSE: " + '{:.5f}'.format(mean_squared_error(y_true=test_target, y_pred=y_predict)))
print("MAE: " + '{:.5f}'.format(mean_absolute_error(y_true=test_target, y_pred=y_predict)))
print("RMSE: " + '{:.5f}'.format(np.sqrt(mean_absolute_error(y_true=test_target, y_pred=y_predict))))
print("MAPE: " + '{:.2f}'.format(mape(y_true=test_target, y_pred=y_predict)) + "%")
print("SMAPE: " + '{:.2f}'.format(smape(y_true=test_target, y_pred=y_predict)) + "%")

