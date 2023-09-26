import matplotlib
import pandas as pd
import numpy as np
import warnings
from sklearn import metrics
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt

matplotlib.rcParams['axes.unicode_minus'] = False
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import scipy.stats as stats
import seaborn as sns

sns.set(font="simhei", style="whitegrid", font_scale=1.6, palette="muted")

warnings.filterwarnings("ignore")
# 读取数据
data = pd.read_csv(r"F:\作业\研究生\数学建模\mathematical-modeling-group\models\data\2-a\class2.txt", sep=' ',
                   encoding='utf-8')

# # 筛选时间小于2000小时并且体积小于150小时
# data = data[(data['Time']<2000) & (data['volume']<150)]

# 创建回归方程
slope, intercept, r_value, p_value, std_err = stats.linregress(data['Time'], data['volume'])

x = data['Time']
y = data['volume']
y_predict = slope * x + intercept

data['Linear_Residuals'] = y_predict - y
id = data.groupby('ID')
grouped_idx = id.groups.keys()
idx_list = list(grouped_idx)
df = pd.DataFrame(idx_list)
residuals = data.groupby('ID').mean()['Linear_Residuals']
print(residuals)

plt.figure(figsize=(8, 6))
group = data.groupby("ID")
for key, value in group:
    x = value["Time"].values
    y = value["volume"].values
    plt.scatter(x, y)

x = data['Time']
y = slope * x + intercept
plt.plot(x, y, color='red', label='Regression Line')
plt.xlabel('Time')
plt.ylabel('Volume')
# plt.xlim(-20, 5000)
# plt.ylim(-5, 180)
plt.tight_layout()  # 自动调整子图参数
# plt.savefig(f'线性回归散点图.png', dpi=500)
plt.show()

