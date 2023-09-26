import numpy as np
import pandas as pd
import statsmodels.api as sm

data = pd.read_excel(r"F:\作业\研究生\数学建模\mathematical-modeling-group\models\data\2-d\train2-4.xlsx")

# 构建模型
model = sm.OLS(data['ED_volume'], sm.add_constant(data[[col for col in data.columns if col not in ['ED_volume']]]))

# 进行拟合
results = model.fit()

# 打印结果
print(results.summary())