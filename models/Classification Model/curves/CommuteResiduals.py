

# 计算残差
def CommuteRe(data, y_true, y_predict):
    data['Linear_Residuals'] = y_predict - y_true
    residuals = data.groupby('ID').mean()['Linear_Residuals']
    return residuals