import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import pandas as pd
import CommuteResiduals as CR

# 读取数据
data = pd.read_csv(r"F:\作业\研究生\数学建模\mathematical-modeling-group\models\data\2-a\class2.txt", sep=' ',
                   encoding='utf-8')

# # 筛选时间小于2000小时并且体积小于150小时
# data = data[(data['Time']<2000) & (data['volume']<150)]

# 多项式回归
poly_reg = Pipeline([
    ('Poly', PolynomialFeatures(degree=2)),
    ('std_scaler', StandardScaler()),
    ('lin_reg', LinearRegression())
])

x = data['Time'].values.reshape(-1, 1)      # 时间间隔
y = data['volume'].values                   # 水肿体积

# 训练预测
poly_reg.fit(x, y)
y_predict = poly_reg.predict(x)

# 按x值进行排序，y值也按照相应顺序进行排序
x_sorted = np.sort(x[:, 0])
y_predict_sorted = y_predict[np.argsort(x[:, 0])]

# 多项式回归方程各系数
a = poly_reg[2].coef_[2]
b = poly_reg[2].coef_[1]
c = poly_reg[2].intercept_

# 计算残差
r = CR.CommuteRe(data, y_true=y, y_predict=y_predict)

print("多项式回归模型， y = {}x**2 + {}x + {}.".format(a, b, c))
plt.figure(figsize=(8, 6))
group = data.groupby("ID")
for key, value in group:
    x = value["Time"].values
    y = value["volume"].values
    plt.scatter(x, y)

plt.plot(x_sorted, y_predict_sorted, color='red', label='Regression Line')
plt.xlabel('Time')
plt.ylabel('Volume')
# plt.xlim(-10, 5000)
# plt.ylim(-5, 150)
plt.tight_layout()  # 自动调整子图参数
plt.savefig(f'2次多项式回归散点图.png', dpi=500)
plt.show()

# data['Linear_Residuals'] = y_predict - y
# id = data.groupby('ID')
# grouped_idx = id.groups.keys()
# idx_list = list(grouped_idx)
# df = pd.DataFrame(idx_list)
# residuals = data.groupby('ID').mean()['Linear_Residuals']
# print(residuals)



