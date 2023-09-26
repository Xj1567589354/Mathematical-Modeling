import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau


# 灰度相关系数计算
def greyanalysis(_other_columns , _mrs_column):
    # 计算关联系数
    correlation_scores = []
    for column in other_columns:
        correlation, _ = pearsonr(mrs_column, other_columns[column])
        correlation_scores.append(correlation)
    return correlation_scores


# Spearmanr相关系数计算
def spearmanranalysis(_other_columns , _mrs_column):
    # 计算关联系数
    correlation_scores = []
    for column in other_columns:
        correlation, _ = spearmanr(mrs_column, other_columns[column])
        correlation_scores.append(correlation)
    return correlation_scores

# Spearmanr相关系数计算
def kendalltauanalysis(_other_columns , _mrs_column):
    # 计算关联系数
    correlation_scores = []
    for column in other_columns:
        correlation, _ = kendalltau(mrs_column, other_columns[column])
        correlation_scores.append(correlation)
    return correlation_scores

# 假设你有一个包含mRS指标和其他指标的数据表格，可以将其导入为一个DataFrame对象
data = pd.read_excel(r"F:\作业\研究生\数学建模\mathematical-modeling-group\models\data\2-c\train23.xlsx")

# 提取mRS指标列和其他感兴趣的指标列
mrs_column = data['ED_volumn_1']
# 获取除了某两列的其他列名
other_columns = data[[col for col in data.columns if col not in ['ED_volume', 'ED_volumn_1']]]

correlation_scores = greyanalysis(other_columns, mrs_column)
# 关联度结果排序
result1 = pd.DataFrame({'Indicator': other_columns.columns, 'Correlation Score': correlation_scores})
# result1 = result1.sort_values(by='Correlation Score', ascending=False)

correlation_scores2 = spearmanranalysis(other_columns, mrs_column)
# 关联度结果排序
result2 = pd.DataFrame({'Indicator': other_columns.columns, 'Correlation Score': correlation_scores2})
# result2 = result2.sort_values(by='Correlation Score', ascending=False)

correlation_scores3 = kendalltauanalysis(other_columns, mrs_column)
# 关联度结果排序
result3 = pd.DataFrame({'Indicator': other_columns.columns, 'Correlation Score': correlation_scores3})
# result3 = result2.sort_values(by='Correlation Score', ascending=False)

print("s")