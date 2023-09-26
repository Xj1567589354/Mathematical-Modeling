import scipy.stats as stats
import pandas as pd

data = pd.read_excel(r"F:\作业\研究生\数学建模\mathematical-modeling-group\models\data\2-d\train2-4.xlsx")

# 治疗方法数据
# treat_data = data[[col for col in data.columns if col not in ['HM_volume', 'ED_volume']]]

# 血肿水肿数据
ed_volume_data = data['ED_volume']
hm_volume_data = data['HM_volume']


# 标准化（Z-Score）归一化
def standardized(_data):
    data = (_data - _data.min()) / (_data.max() - _data.min())
    return data

# standardized_treat_data= (treat_data - treat_data.min()) / (treat_data.max() - treat_data.min())
# standardized_volume_data = (volume_data - volume_data.min()) / (volume_data.max() - volume_data.min())
ed_volume_data_standard = standardized(ed_volume_data)
hm_volume_data_standard = standardized(hm_volume_data)

df = pd.DataFrame({'ID': [], 'F': [], 'P': []})
# for i in ed_volume_data_standard.columns:
#     data = standardized_treat_data[i]
#     # 将数据分成两组
#     group_0 = standardized_volume_data[data == 0]
#     group_1 = standardized_volume_data[data == 1]
#     f_statistic, p_value = stats.f_oneway(group_0, group_1)
#
#     a = {'ID': [i], 'F': [f_statistic], 'P': [p_value]}
#     # 将数据添加到 DataFrame
#     new_data = pd.DataFrame(a)
#     df = pd.concat([df, new_data], ignore_index=True)

f_statistic, p_value = stats.f_oneway(ed_volume_data_standard, hm_volume_data_standard)
print("s")

# df.to_excel('data3.xlsx', index=False)
