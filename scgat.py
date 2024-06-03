import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GAE
from torch_geometric.data import Data
from scipy.stats import pearsonr, spearmanr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_features = 852
EmbeddingDim = 64
out_channels = 64
learning_rate = 5e-3
num_clusters = 6

def generate_mask_sets(j):
    seed = j
    num_rows = 10
    num_cols = 270
    num_sets = 10
    torch.manual_seed(seed)
    missing_ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]

    def generate_indices(num_rows, num_cols, missing_ratios):
        all_cols_indices = torch.randperm(num_cols)
        mask = torch.zeros(num_rows, num_cols, dtype=torch.bool)

        for i, ratio in enumerate(missing_ratios):
            start_col = int(num_cols * 0.1 * i)
            end_col = int(num_cols * 0.1 * (i + 1))
            num_missing_edges_per_col = int(num_rows * ratio)
            for col in range(start_col, end_col):
                selected_rows = torch.randperm(num_rows)[:num_missing_edges_per_col]
                mask[selected_rows, all_cols_indices[col]] = True

        flat_indices = torch.nonzero(mask.flatten()).squeeze()
        all_indices = torch.arange(num_rows * num_cols, dtype=torch.long)
        remaining_indices = torch.tensor([idx.item() for idx in all_indices if idx not in flat_indices], dtype=torch.long)
        return flat_indices, remaining_indices, mask

    all_train_indices = []
    all_test_indices = []
    for _ in range(num_sets):
        train_indices, test_indices, mask = generate_indices(num_rows, num_cols, missing_ratios)
        all_train_indices.append(train_indices)
        all_test_indices.append(test_indices)
    return all_train_indices[j], all_test_indices[j] 

source_embedding = np.load("./DomainNet_domain_source_embedding.npy")
target_embedding = torch.tensor(np.load("./DomainNet_domain_target_embedding.npy")).float()
task_embedding = torch.tensor(np.concatenate((source_embedding, target_embedding))).float().to(device)
t_affinity_matrix_tensor = pd.read_csv("./DomainNet_10_270_gd.csv", index_col=0)
t_affinity_matrix = t_affinity_matrix_tensor.values

edge_index = []
for i in range(t_affinity_matrix.shape[0]):
    for j in range(t_affinity_matrix.shape[0], t_affinity_matrix.shape[0] + t_affinity_matrix.shape[1]):
        edge_index.append([i, j])
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        hidden_channels = 64
        self.l_in = torch.nn.Linear(in_channels, hidden_channels)
        self.l_out = torch.nn.Linear(hidden_channels * 3, out_channels)
        self.conv1 = GATConv(hidden_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)

    def forward(self, x, ei, ew):
        x1 = self.l_in(x)
        x2 = self.conv1(x1, ei, ew)
        x3 = self.conv2(x2, ei, ew)
        x_all = torch.cat([x1, x2, x3], dim=1)
        x = self.l_out(x_all)
        return x

class LinearDecoder(torch.nn.Module):
    def __init__(self):
        super(LinearDecoder, self).__init__()
        self.decoder = torch.nn.Linear(2 * out_channels, 1)

    def forward(self, z):
        z_src = z[:10, :]
        z_dst = z[10:, :]
        z_concat = torch.cat([z_src.repeat_interleave(z_dst.size(0), dim=0),
                              z_dst.repeat(z_src.size(0), 1)], dim=1)
        dot_product = self.decoder(z_concat)
        return torch.sigmoid(dot_product)

results = []
for n_cross in range(10):
    result = {}
    train_edge_indices, test_edge_indices = generate_mask_sets(n_cross)

    edge_weight = torch.Tensor(t_affinity_matrix).to(device)
    train_edge_index = edge_index[:, train_edge_indices]
    test_edge_index = edge_index[:, test_edge_indices]

    train_edge_weight = edge_weight.reshape(-1)[train_edge_indices]
    test_edge_weight = edge_weight.reshape(-1)[test_edge_indices]

    train_data = Data(x=task_embedding, edge_index=train_edge_index, edge_attr=train_edge_weight).to(device)
    test_data = Data(x=task_embedding, edge_index=test_edge_index, edge_attr=test_edge_weight).to(device)

    test_ground_truth = train_label = edge_weight

    encoder = GCNEncoder(num_features, out_channels)
    decoder = LinearDecoder()
    model = GAE(encoder, decoder).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index, train_data.edge_attr)
        out = model.decode(z)

        train_out = out.reshape(-1)[train_edge_indices]
        loss = criterion(train_out.view(-1), train_edge_weight.view(-1))
        loss.backward()
        optimizer.step()

        train_input_matrix = torch.zeros(2700, device=device)
        train_input_matrix[train_edge_indices] = train_data.edge_attr
        train_output_matrix = out.view(10, 270).detach().cpu().numpy()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                z = model.encode(test_data.x, train_data.edge_index, train_data.edge_attr)
                out = model.decode(z)
                out = out.reshape(-1)
                masked_out = test_ground_truth.clone().reshape(-1)
                masked_out[test_edge_indices] = out[test_edge_indices]
                masked_out = masked_out.reshape(10, 270)

                spearman_corrs = []
                for i in range(masked_out.shape[1]):
                    spearman_corr = pearsonr(test_ground_truth[:, i].detach().cpu().numpy(),
                                            masked_out[:, i].detach().cpu().numpy()).correlation
                    spearman_corrs.append(spearman_corr)
                mean_spearman_corr = np.nanmean(spearman_corrs)

            test_input_matrix = torch.zeros(2700, device=device)
            test_input_matrix[train_edge_indices] = train_data.edge_attr

            test_input_matrix = test_input_matrix.detach().cpu().numpy()
            test_output_matrix = masked_out.detach().cpu().numpy()

            print(f"Cluster: {n_cross}, Epoch: {epoch}, Train Loss: {loss:.4f}, Test Spearman Correlation: {mean_spearman_corr:.4f}")

            if epoch + 1 == 300:
                result["test"] = round(mean_spearman_corr, 4)

    def get_sorted_indices_by_zero_count(matrix):
        zero_counts = np.sum(matrix == 0, axis=0)
        sorted_indices = np.argsort(-zero_counts)
        return sorted_indices

    test_input_matrix = torch.zeros(2700, device=device)
    test_input_matrix[train_edge_indices] = train_data.edge_attr
    test_input_matrix = test_input_matrix.detach().cpu().numpy()
    sorted_indices = get_sorted_indices_by_zero_count(test_input_matrix.reshape(10, 270))
    sorted_test_ground_truth = test_ground_truth.cpu().numpy()[:, sorted_indices]
    sorted_test_output_matrix = masked_out[:, sorted_indices]

    thresholds = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    num_columns = 270

    for i in range(len(thresholds) - 1):
        lower_threshold = thresholds[i]
        upper_threshold = thresholds[i + 1]
        lower_index = int(lower_threshold * num_columns)
        upper_index = int(upper_threshold * num_columns)
        
        spearman_corrs = []
        for j in range(lower_index, upper_index):
            spearman_corr = spearmanr(sorted_test_ground_truth[:, j], sorted_test_output_matrix[:, j].cpu().numpy()).correlation
            spearman_corrs.append(spearman_corr)
        
        mean_spearman_corr = np.mean(spearman_corrs)
        result[f"{lower_index}-{upper_index}"] = round(mean_spearman_corr, 4)

    print(result)
    results.append(result)

df = pd.DataFrame(results)
mean_row = df.mean()
std_row = df.std()
df.loc['mean'] = mean_row
df.loc['std'] = std_row
df.to_csv(f"result.csv")
