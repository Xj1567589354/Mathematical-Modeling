import pandas as pd
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# 一阶函数方程(直线)
def func_1(x, a, b):
    return a*x + b


# 二阶曲线方程
def func_2(x, a, b, c):
    return a * np.power(x, 2) + b * x + c


# 四阶曲线方程
def func_4(x, a, b, c, d, e):
    return a * np.power(x, 4) + b * np.power(x, 3) + c * np.power(x, 2) + d * x + e


data = pd.read_csv(r"F:\作业\研究生\数学建模\mathematical-modeling-group\models\data\2-a\time_volume.txt", sep=' ', encoding='utf-8')

time_data = data['Time'].values
volume_data = data['volume'].values
# 假设您的数据集包含自变量 x 和因变量 y
# time_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
# volume_data = np.array([0.2, 0.3, 0.5, 0.8, 1.0])

# 拟合参数都放在popt里，popt是个数组，参数顺序即你自定义函数中传入的参数的顺序
popt, pcov= curve_fit(func_4, time_data, volume_data)

# 获取拟合参数
a_fit = popt[0]
b_fit = popt[1]
c_fit = popt[2]
d_fit = popt[3]
e_fit = popt[4]

# 生成一系列输入值（用于绘制拟合曲线）
x_fit = time_data
y_fit = func_4(x_fit, a_fit, b_fit, c_fit, d_fit, e_fit)

# 绘制原始数据和拟合曲线
plt.scatter(time_data, volume_data, color='red', label='Data')
plt.plot(x_fit, y_fit, color='blue', label='Gaussian Fit')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gaussian Fit')
plt.legend()
plt.show()



