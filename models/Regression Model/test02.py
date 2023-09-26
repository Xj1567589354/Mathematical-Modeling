import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

# 加载数据集
data = pd.read_csv(r'F:\作业\研究生\数学建模\mathematical-modeling-group\models\data\train_T.txt', sep='\t',  encoding='utf-8')

# 提取特征和目标变量
X = data.drop('target', axis=1)
y = data['target']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 实例化特征处理器
scaler = StandardScaler()
# 对训练集进行标准化
X_train_scaled = scaler.fit_transform(X_train)
# 对测试集进行相同的标准化处理
X_test_scaled = scaler.transform(X_test)

# 划分训练集为训练集和验证集
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)


# 设置参数
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,  # 设置verbosity为-1，以隐藏警告信息
}

# 实例化模型
model = LGBMClassifier()
model.set_params(**params)
# 模型训练
model.fit(X_train_final, y_train_final)
# 在验证集上进行预测
val_predictions = model.predict(X_val)
# 计算验证集上的准确率
val_accuracy = accuracy_score(y_val, val_predictions)
print(val_accuracy)


# 对测试集进行预测
test_predictions = model.predict(X_test_scaled)
# 计算测试集上的准确率
test_accuracy = accuracy_score(y_test, test_predictions)
print(test_accuracy)









