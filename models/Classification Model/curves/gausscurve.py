import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import CommuteResiduals as CR


# 读取数据
data = pd.read_csv(r"F:\作业\研究生\数学建模\mathematical-modeling-group\models\data\2-a\class2.txt", sep=' ',
                   encoding='utf-8')

# # 筛选时间小于2000小时并且体积小于150小时
# data = data[(data['Time']<2000) & (data['volume']<150)]

x = data['Time'].values      # 时间间隔

x_shape = data['Time'].values.reshape(-1, 1)
y = data['volume'].values                   # 水肿体积


# 定义高斯函数
def gaussian_func(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2 / (2 * stddev ** 2)))


# y_noise = gaussian_func(x, 1, 0, 1) + np.random.normal(0, 0.05, 440)  # 加入噪声


# 进行高斯拟合训练预测
popt, pcov = curve_fit(gaussian_func, x, y, maxfev=10000)
y_predict = gaussian_func(x, *popt)

# 按x值进行排序，y值也按照相应顺序进行排序
x_sorted = np.sort(x_shape[:, 0])
y_predict_sorted = y_predict[np.argsort(x_shape[:, 0])]

# 计算残差
r = CR.CommuteRe(data, y_true=y, y_predict=y_predict)

plt.figure(figsize=(8, 6))
group = data.groupby("ID")
for key, value in group:
    x = value["Time"].values
    y = value["volume"].values
    plt.scatter(x, y)

plt.plot(x_sorted, y_predict_sorted, color='red', label='Regression Line')
plt.xlabel('Time')
plt.ylabel('Volume')
# plt.xlim(-10, 2000)
# plt.ylim(-5, 150)
plt.tight_layout()  # 自动调整子图参数
plt.savefig(f'高斯拟合散点图.png', dpi=500)
plt.show()


