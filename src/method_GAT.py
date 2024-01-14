#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/11/5 16:18
# @Author : Luo Yong(MGYL)
# @File : testGAT1105.py
# @Software: PyCharm
import torch
import numpy as np
import pandas as pd
import networkx as nx
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score


# 定义 GAT 模型
class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, num_classes, heads=1)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


path = "../data/moviesData/ml_1m/stageA/"

net = pd.read_csv(path + "net.csv")

# 创建一个空的无向图
G = nx.Graph()
# 添加连边到图
edges = list(net[['id_x', 'id_y']].itertuples(index=False, name=None))
G.add_edges_from(edges)

# 邻接矩阵
adjacency_matrix = nx.adjacency_matrix(G)
edge_index = torch.LongTensor(np.array(G.edges()).T)

kfold = KFold(n_splits=5, shuffle=True, random_state=100)

GATList = list()
for q in range(5, 55, 5):
    print("感知用户Top：", q)
    # 特征数据集
    idx_feature_label = pd.read_csv(path + "idx_feature_label_" + str(q) + ".csv")
    # 提取数据集ID索引
    data = list(idx_feature_label['id'].values)
    # 将节点分为训练集和测试集
    train_index, test_index = train_test_split(data, test_size=0.3, random_state=42)

    input_features = torch.tensor(idx_feature_label[['degree_normalized', 'reputation', 'dfa']].values,
                                  dtype=torch.float32)
    labels = torch.LongTensor(idx_feature_label[['isHPU']].values).view(-1)

    gatList = list()
    j = 0
    for train_idx, val_idx in kfold.split(data):
        j += 1
        # 为训练和测试节点创建标签
        train_labels = labels[train_idx]
        test_labels = labels[val_idx]
        # 创建 GAT 模型
        model = GAT(in_channels=input_features.shape[1], hidden_channels=8, num_classes=2, heads=2)

        # 定义损失函数和优化器
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005)  # 0.01

        # 训练 GAT 模型
        for epoch in range(500):
            output = model(input_features, edge_index)

            train_output = output[train_idx]
            onehot_target = torch.eye(2)[train_labels.long(), :]
            train_loss = criterion(train_output, onehot_target)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch + 1}/100], Train Loss: {train_loss.item()}')

        # 在测试集上评估模型
        test_output = output[val_idx]
        predicted_labels = torch.argmax(test_output, dim=1)
        true_labels = labels[val_idx]
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
GATList.to_csv(path + '/rs_methGAT20231227.csv', index=False, encoding="utf_8_sig")
