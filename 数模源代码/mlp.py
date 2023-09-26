import torch
import torch.nn as nn
import pandas as pd
from sklearn import preprocessing
import Train_Predict as T
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim


# 定义 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 特征归一化
def standize_df(train_data, test_data):
    features_columns = [col for col in train_data.columns if col not in ['target']]
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler = min_max_scaler.fit(train_data[features_columns])

    train_data_scaler = min_max_scaler.transform(train_data[features_columns])
    test_data_scaler = min_max_scaler.transform(test_data[features_columns])

    train_data_scaler = pd.DataFrame(train_data_scaler)
    train_data_scaler.columns = features_columns

    test_data_scaler = pd.DataFrame(test_data_scaler)
    test_data_scaler.columns = features_columns

    train_data_scaler['target'] = train_data['target']
    return train_data_scaler, test_data_scaler


train_f3 = pd.read_csv(r"F:\作业\研究生\数学建模\mathematical-modeling-group\models\data\train.txt", sep='\t', encoding='utf-8')
test_f3 = pd.read_csv(r"F:\作业\研究生\数学建模\mathematical-modeling-group\models\data\test.txt", sep='\t', encoding='utf-8')
train_f3, test_f3 = standize_df(train_f3, test_f3)

target = T.get_target_df(train_f3).copy()

# 随机划分数据集，20%概率验证集
train_all, val_all, train_target, val_target = T.train_test_split(train_f3, target, test_size=0.2, random_state=0)

train_data = T.get_predictors_df(train_all).copy()
val_data = T.get_predictors_df(val_all).copy()

# 转换为PyTorch张量
train_data = torch.tensor(train_data.values, dtype=torch.float32)
train_target = torch.tensor(train_target.values, dtype=torch.long)
val_data = torch.tensor(val_data.values, dtype=torch.float32)
val_target = torch.tensor(val_target.values, dtype=torch.long)

# # 数据归一化
# mean = train_data.mean(dim=0)
# std = train_data.std(dim=0)

# 定义超参数
input_size = train_data.size(1)
hidden_size = 224
num_classes = 3
num_epochs = 10
batch_size = 32
learning_rate = 0.01

# # 定义输入维度、隐藏层维度和类别数
# input_size = 5840  # MNIST 图像大小为 28x28，将图像展平为 784 维向量
# hidden_size = 256
# num_classes = 2  # MNIST 数据集有 10 个类别：0 到 9

# 创建 MLP 模型实例
model = MLP(input_size, hidden_size, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_batches = (train_data.size(0) - 1) // batch_size + 1
for epoch in range(num_epochs):
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, train_data.size(0))
        batch_features = train_data[start_idx:end_idx]
        batch_labels = train_target[start_idx:end_idx]

        # 前向传播
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 在验证集上计算准确率
    with torch.no_grad():
        valid_outputs = model(val_data)
        _, predicted = torch.max(valid_outputs.data, 1)
        accuracy = (predicted == val_target).sum().item() / val_target.size(0)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Accuracy: {accuracy}')

test_data = torch.tensor(test_f3.values, dtype=torch.float32)

model.eval()
with torch.no_grad():
    test_outputs = model(test_data)
    _, predicted = torch.max(test_outputs.data, 0)
    print("s")

