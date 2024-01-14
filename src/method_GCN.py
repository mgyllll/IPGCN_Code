#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/11/8 10:50
# @Author : Luo Yong(MGYL)
# @File : testGraphSAGE.py
# @Software: PyCharm
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score


# 定义GCN模型
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNLayer(input_dim, hidden_dim)
        self.conv2 = GCNLayer(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, edge_index):
        row, col = edge_index
        neighbor_x = x[col]
        aggr_x = self.linear(neighbor_x)
        return aggr_x


path = "../data/moviesData/douban/"

net = pd.read_csv(path + "net.csv")

# 创建一个空的无向图
G = nx.Graph()
# 添加连边到图
edges = list(net[['id_x', 'id_y']].itertuples(index=False, name=None))
G.add_edges_from(edges)
adjacency_matrix = nx.to_numpy_matrix(G)
# 邻接矩阵
# 创建图的邻接矩阵
adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)
edge_index = adjacency_matrix.nonzero().t()


GATList = list()
for q in range(5, 55, 5):
    print("感知用户Top：", q)
    idx_feature_label = pd.read_csv(path + "idx_feature_label_" + str(q) + ".csv")
    pos_data = list(idx_feature_label[idx_feature_label['isHPU'] == 1]['id'].values)
    neg_data = list(idx_feature_label[idx_feature_label['isHPU'] == 0]['id'].values)
    data = (pos_data + random.sample(neg_data, len(pos_data)*1)) if len(pos_data) * 2 < len(idx_feature_label) else (pos_data +neg_data)

    # 将节点分为训练集和测试集
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

    input_features = torch.tensor(idx_feature_label[['degree_normalized', 'reputation', 'dfa']].values,
                                  dtype=torch.float32)
    labels = torch.LongTensor(idx_feature_label[['isHPU']].values).view(-1)
    # 为训练和测试节点创建标签
    train_labels = labels[train_data]
    test_labels = labels[test_data]

    gatList = list()
    for j in range(10):
        # 创建 GAT 模型
        # 初始化模型和优化器
        model = GCN(input_dim=3, hidden_dim=32, num_classes=2)

        # 定义损失函数和优化器
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)  # 0.01

        # 训练 GAT 模型
        for epoch in range(500):
            output = model(input_features, edge_index)

            train_output = output[train_data]
            onehot_target = torch.eye(2)[train_labels.long(), :]
            train_loss = criterion(train_output, onehot_target)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch + 1}/100], Train Loss: {train_loss.item()}')

        # 在测试集上评估模型
        test_output = output[test_data]
        predicted_labels = torch.argmax(test_output, dim=1)
        true_labels = labels[test_data]
        print("True Labels (测试标签):", true_labels)
        print("Predicted Labels (测试集):", predicted_labels.tolist())
        fpr, tpr, threshold = roc_curve(true_labels, predicted_labels)  # 计算真正率和假正率
        roc_auc = auc(fpr, tpr)  # 计算auc的值
        print("训练", j, "=>", [precision_score(true_labels, predicted_labels),
                              recall_score(true_labels, predicted_labels),
                              f1_score(true_labels, predicted_labels), roc_auc])
        gatList.append([precision_score(true_labels, predicted_labels), recall_score(true_labels, predicted_labels),
                        f1_score(true_labels, predicted_labels), roc_auc])

    gatList = pd.DataFrame(gatList, columns=['precision', 'recall', 'f1', 'auc'])
    GATList.append([q, gatList['precision'].mean(), gatList['recall'].mean(), gatList['f1'].mean(),
                    gatList['auc'].mean()])
    print("感知用户Top", q, "多次训练结果=>", [gatList['precision'].mean(), gatList['recall'].mean(), gatList['f1'].mean(),
                                     gatList['auc'].mean()])
GATList = pd.DataFrame(GATList, columns=['q', 'precision', 'recall', 'f1', 'auc'])
GATList.to_csv(path + '/rs_methodGCN_1000.csv', index=False, encoding="utf_8_sig")
