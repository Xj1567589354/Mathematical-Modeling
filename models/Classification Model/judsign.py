import scipy.stats as stats
import pandas as pd

# 假设你有一个包含mRS指标和其他指标的数据表格，可以将其导入为一个DataFrame对象
data = pd.read_excel(r"F:\作业\研究生\数学建模\mathematical-modeling-group\models\Classification Model\data2.xlsx")
f_value = data['F']
p_value = data['P']

i = 0
for a in range(7):
    if p_value[i] <= 0.01 and f_value[i] > 4.02:
        print("非常显著")
    elif p_value[i] <= 0.05 and f_value[i] > 4.02:
        print("显著")
    else:
        print("不显著")
    i += 1



