import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize as op


data = pd.read_csv(r"F:\作业\研究生\数学建模\mathematical-modeling-group\models\data\2-a\time_volume.txt", sep=' ', encoding='utf-8')

df = pd.DataFrame(columns=['ID', 'A', 'B', 'C', 'D'])

# 按照 ID 列进行分组
grouped = data.groupby('ID')

for key, value in grouped:
    time_data = value.Time
    volume_data = value.volume
    z1 = np.polyfit(time_data, volume_data, 3)
    new_row = {'ID': key, 'A': z1[0], 'B': z1[1], 'C': z1[2], 'D': z1[3]}
    df = pd.concat([df, pd.DataFrame(new_row, index=[0])], ignore_index=True)

print("s")

# # 提供的数据
# time_data = data.iloc[0:5, 1:2]
# volume_data = data.iloc[0:5, 2:]
#
# time_data = time_data['Time'].tolist()
# volume_data = volume_data['volume'].tolist()

z1 = np.polyfit(time_data, volume_data, 3)
p1 = np.poly1d(z1)
y_pre = p1(time_data)

plt.plot(time_data, volume_data, '.')
plt.plot(time_data, y_pre)
plt.show()
