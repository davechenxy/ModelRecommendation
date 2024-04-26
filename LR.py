import torch
import numpy as np
from scipy.stats import spearmanr
import numpy as np
import torch
import torch.nn as nn

source_embedding = np.load("./Embedding/Model_embedding.npy")
target_embedding = torch.tensor(np.load("./Embedding/Dataset_embedding.npy")).float()
task_embedding = torch.tensor(np.concatenate((source_embedding, target_embedding))).float()
t_affinity_matrix_tensor = pd.read_csv("./record_domainnet_10_270.csv",index_col=0)
t_affinity_matrix = t_affinity_matrix_tensor.values

m = 10
s = 3
d = 6
t = 15

edge_weight = np.arange(m*s*d*t).reshape([m, s, d, t])

cross_domain_index = {}
for i in range(d):
    test_index = edge_weight[:, :, i:i+1, :].flatten()
    train_index = np.concatenate([edge_weight[:, :, 0:i, :], edge_weight[:, :, i+1:d, :]], axis=2).flatten()
    print(i, len(test_index), len(train_index))
    cross_domain_index[f"cross{i}"] = {
        "train_index": train_index,
        "test_index": test_index
    }

def split_data(node_embedding, edge_index, edge_weight, mask_ratio=0.5):
    # # 随机打乱边的索引
    # num_edges = edge_index.shape[1]
    # permuted_indices = torch.randperm(num_edges)

    # # 划分训练集和测试集的边索引
    # num_train_edges = int(num_edges * (1 - mask_ratio))
    # train_edge_indices = permuted_indices[:num_train_edges]
    # test_edge_indices = permuted_indices[num_train_edges:]

    train_edge_indices = cross_domain_index["cross1"]["train_index"]
    test_edge_indices = cross_domain_index["cross1"]["test_index"]

    # 划分边索引
    train_edge_index = edge_index[:, train_edge_indices]
    test_edge_index = edge_index[:, test_edge_indices]

    # 获取训练和测试边的权重
    train_edge_weight = edge_weight.reshape(-1)[train_edge_indices]
    test_edge_weight = edge_weight.reshape(-1)[test_edge_indices]

    # 创建训练集和测试集的Data对象
    train_data = Data(x=node_embedding, edge_index=train_edge_index, edge_attr=train_edge_weight)
    test_data = Data(x=node_embedding, edge_index=test_edge_index, edge_attr=train_edge_weight)

    train_label = edge_weight

    return train_data, train_label, test_data, train_edge_indices, test_edge_indices

edge_index = []
for i in range(t_affinity_matrix.shape[0]):
    for j in range(t_affinity_matrix.shape[0],t_affinity_matrix.shape[0] + t_affinity_matrix.shape[1]):
        edge_index.append([i, j])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

train_data, train_label, test_data, train_edge_indices, test_edge_indices = split_data(task_embedding, edge_index, torch.Tensor(t_affinity_matrix))
test_ground_truth = torch.Tensor(t_affinity_matrix).to(device)
test_mask_indices_matrix = [(y, x) for x, y in zip(test_data.edge_index[1] % 270, test_data.edge_index[0])]

class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# 准备线性回归的输入和输出
X_train = []
y_train = []
for i in range(train_data.edge_index.shape[1]):
    src, dst = train_data.edge_index[0, i].item(), train_data.edge_index[1, i].item()
    X_train.append(np.concatenate([train_data.x[src], train_data.x[dst]]))
    y_train.append(train_data.edge_attr[i].item())

X_test = []
y_test = []
for i in range(test_data.edge_index.shape[1]):
    src, dst = test_data.edge_index[0, i].item(), test_data.edge_index[1, i].item()
    X_test.append(np.concatenate([test_data.x[src], test_data.x[dst]]))
    y_test.append(test_data.edge_attr[i].item())


# 准备数据
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# 创建线性回归模型
input_dim = X_train_tensor.shape[1]
model = LinearRegression(644)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 训练模型
num_epochs = 5000
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # 在测试集上评估模型
        with torch.no_grad():
            outputs = model(X_test_tensor)
            predicted = outputs.numpy()
            mse = criterion(outputs, y_test_tensor)
            print(f'Mean Squared Error: {mse.item():.4f}')


        # 创建预测值矩阵
        y_pred_matrix = np.zeros((t_affinity_matrix.shape[0], t_affinity_matrix.shape[1]))

        # 使用 test_mask_indices_matrix 填充预测值
        for i, (row, col) in enumerate(test_mask_indices_matrix):
            y_pred_matrix[row, col] = predicted[i]

        # 创建真实值矩阵
        y_test_matrix = test_ground_truth

        # 创建测试集的掩码
        test_mask = np.zeros((t_affinity_matrix.shape[0], t_affinity_matrix.shape[1]), dtype=bool)
        for row, col in test_mask_indices_matrix:
            test_mask[row, col] = True

        # 将未被掩码的部分替换为真实值
        masked_y_pred_matrix = y_pred_matrix.copy()
        masked_y_pred_matrix[~test_mask] = y_test_matrix[~test_mask]

        # 计算列相关性
        spearman_corrs_lr = []
        for i in range(masked_y_pred_matrix.shape[1]):
            spearman_corr = spearmanr(y_test_matrix[:, i], masked_y_pred_matrix[:, i]).correlation
            spearman_corrs_lr.append(spearman_corr)

        mean_spearman_corr_lr = np.nanmean(spearman_corrs_lr)
        print(f"Linear Regression - Average Test Spearman Correlation: {mean_spearman_corr_lr:.4f}")