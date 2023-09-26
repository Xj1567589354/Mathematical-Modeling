import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.datasets import load_iris

# 读取数据
iris = load_iris()
x = iris.data[:, 2:]
print(x[:, 0])
print(x[:, 1])

# # 绘制数据分布图
# plt.scatter(x[:, 0], x[:, 1], c = "red", marker='o', label='see')
# plt.xlabel('petal length')
# plt.ylabel('petal width')
# plt.legend(loc=2)
# plt.show()

estimator = KMeans(n_clusters=3)    # 构造聚类器
estimator.fit(x)                    # 训练
label_pred = estimator.labels_      # 获取聚类标签

# 绘制k-means结果
# 前三类标签
x0 = x[label_pred == 0]
x1 = x[label_pred == 1]
x2 = x[label_pred == 2]
plt.scatter(x0[:, 0], x0[:, 1], c = "red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c = "green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c = "blue", marker='+', label='label2')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()

