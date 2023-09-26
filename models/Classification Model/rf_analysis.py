import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

plt.rcParams['font.sans-serif'] = ['SimHei']

data = pd.read_excel(r"F:\作业\研究生\数学建模\mathematical-modeling-group\models\data\2-c\train23.xlsx")

# 随机森林检查特征重要性排序

x = data[[col for col in data.columns if col not in ['ED_volume', 'ED_volumn_1']]]   # 其他特征
y = data['ED_volumn_1']  # target

feat_labels = data.iloc[:, :-1].columns

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

forest = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1, max_depth=3)
forest.fit(x_train, y_train.astype('int'))
score = forest.score(x_test, y_test.astype('int'))

importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]  # 下标排序
for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))