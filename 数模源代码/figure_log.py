import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_excel(r'F:\作业\研究生\数学建模\mathematical-modeling-group\models\data\train33.xlsx')

correlation_matrix = data.corr()  # 计算DataFrame的相关系数矩阵
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
