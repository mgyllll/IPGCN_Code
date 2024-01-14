#!/usr/bin/env python
# coding: utf-8

"""
Python    : 3.7
@File     : Utils.py
@Copyright: MGYL
@Author   : LuoYong
@Date     : 2022-09-07
@Desc     : NULL
"""
import math
import torch
import random
import numpy as np
import pandas as pd
import networkx as nx
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def generate_subgraph(G, node_list):
    """根据目标节点获取邻接矩阵
    Parameters:
        G:目标网络
        node_list:包含目标节点和固定数量邻居节点的列表
    return:
        G_sub: 子网络
        A: 对应的邻居网络
    """
    G_sub = nx.Graph()
    L = len(node_list)
    encode = dict(zip(node_list, list(range(L))))  # 对节点进行重新编号
    subgraph = nx.subgraph(G, node_list)  # 提取子图
    subgraph_edges = list(subgraph.edges())  # 获取子图的边列表
    new_subgraph_edges = []
    for i, j in subgraph_edges:
        new_subgraph_edges.append((encode[i], encode[j]))
    G_sub.add_edges_from(new_subgraph_edges)
    A = np.zeros([L, L])
    for i in range(L):
        for j in range(L):
            if G_sub.has_edge(i, j) and (i != j):
                A[i, j] = 1
    return G_sub, A


def transform(A, degree_list, rep_list):
    """按规则进行转换
    Parameters:
        A: 邻接矩阵
        degree_list:所选节点对应的度值
    return:
        B:单通道的嵌入矩阵
    """
    B = A
    B[0, 1:] = A[0, 1:] * (np.array(rep_list)[1:])
    B[1:, 0] = A[1:, 0] * (np.array(rep_list)[1:])
    for i in range(len(rep_list)):
        B[i, i] = rep_list[i]
    return B


# 用户评分序列和评分习惯编码
def embeddings(df1, df2, df3, L):
    data_dict = {}
    for u in tqdm(set(np.array(df1['user_id']))):

        tmp_df = df1[df1['user_id'] == u].copy(deep=True)
        tmp_df['time'] = tmp_df.apply(lambda a: -a['timestamp'], axis=1)
        tmp_df = tmp_df.sort_values(by=['rating', 'time'], axis=0, ascending=False)
        l = (L - 1) if (L - 1) < len(tmp_df) else len(tmp_df)
        movies_u = np.array(tmp_df['movie_id'].head(l))
        qty_m = list()  # RBeta电影的质量
        degree_m = list()  # 电影的度值
        for m in movies_u:
            qty_m.append(float(df2[df2['movie_id'] == m]['quality']))
            degree_m.append(len(df1[df1['movie_id'] == m]))
        A = np.zeros([L, L])
        A[0, 0] = df3[df3['user_id'] == u]['dfa']  # 用户DFA值
        ur = np.array(tmp_df['rating'].head(l))
        A[0, 1:l + 1] = ur
        A[1:l + 1, 0] = ur
        for i in range(1, l + 1):
            A[i, i] = qty_m[i - 1]
        for r in range(1, l + 1):
            for s in range(r + 1, l + 1):
                A[r, s] = A[s, r] = 1 / math.exp(abs(abs(ur[r - 1] - ur[s - 1]) - abs(qty_m[r - 1] - qty_m[s - 1])))
        data_dict[u] = A
    return data_dict


def embeddings1(df1, df2, df3, L):
    # 未加入打分习惯
    data_dict = {}
    for u in tqdm(set(np.array(df1['user_id']))):
        tmp_df = df1[df1['user_id'] == u].copy(deep=True)
        tmp_df['time'] = tmp_df.apply(lambda a: -a['timestamp'], axis=1)
        tmp_df = tmp_df.sort_values(by=['rating', 'time'], axis=0, ascending=False)
        l = (L - 1) if (L - 1) < len(tmp_df) else len(tmp_df)
        movies_u = np.array(tmp_df['movie_id'].head(l))
        qty_m = list()  # RBeta电影的质量
        degree_m = list()  # 电影的度值
        for m in movies_u:
            qty_m.append(float(df2[df2['movie_id'] == m]['Q']))
            degree_m.append(len(df1[df1['movie_id'] == m]))
        A = np.zeros([L, L])
        A[0, 0] = df3[df3['user_id'] == u]['dfa']  # 用户DFA值
        ur = np.array(tmp_df['rating'].head(l))
        A[0, 1:l + 1] = ur
        A[1:l + 1, 0] = ur
        data_dict[u] = A
    return data_dict


# 加入用户相似网络的声誉reputation
def embeddings2(G, L, rep_dict):
    """节点输入构造主程序
    Parameters:
        G: 目标网络
        L: 嵌入矩阵的大小（包含目标节点的邻居网络节点总数）
    return:
        data_dict:存储每个节点嵌入矩阵的字典{v1:matrix_v1,...,v2:matrix_v2,...}
    """
    data_dict = {}
    node_list = list(G.nodes())  # 获得网络中的所有节点
    # 对每个节点按照规则提取L-1个邻居节点
    for node in tqdm(node_list):
        subset = [node]  # 目标节点+固定数量的邻居节点
        one_order = list(G.adj[node])  # 先看一阶邻居节点
        one_degree = dict(G.degree(one_order))  # 获取一阶邻居节点的度值
        if len(one_order) >= L - 1:  # 如果一阶邻居节点够了，那就不看二阶邻居了
            selected_degree = [len(one_order)]  # 所选节点在原始网络的邻居数量
            selected_rep = [rep_dict[node]]

            selected_nei = [i for i, j in sorted(one_degree.items(), key=lambda x: x[1], reverse=True)]  # 按度值对一阶邻居排序
            for nei in selected_nei:
                if (nei not in subset) and (len(subset) < L):
                    subset.append(nei)
                    selected_degree.append(one_degree[nei])
                    selected_rep.append(rep_dict[nei])
            node_subgraph, node_A = generate_subgraph(G, subset)  # 生成阶邻矩阵
            node_B = transform(node_A, selected_degree, selected_rep)  # 转换
            data_dict[node] = node_B

        elif (len(one_order) < L - 1) and (len(one_order) != 0):  # 当一阶邻居节点不够并且一阶邻居数量不为0的时候，找更高阶的邻居
            selected_degree = [len(one_order)]
            selected_rep = [rep_dict[node]]

            selected_nei = [i for i, j in sorted(one_degree.items(), key=lambda x: x[1], reverse=True)]
            gap = (L - 1) - len(selected_nei)  # 看看还差多少
            high_nei = set(selected_nei)  # 高阶邻居节点
            neis = selected_nei
            count = 0  # 尝试50次，如果超过了50次就用padding
            while True:
                if count == 50:
                    break
                new_order = set([])
                for nei in neis:  # 遍历每个邻居节点的邻居
                    nei_nei = list(G.adj[nei])
                    for each in nei_nei:
                        if (each != node) and (each not in high_nei):
                            new_order.add(each)
                new_order_list = list(new_order)
                degree_new = dict(G.degree(new_order_list))
                new_selected_nei = [i for i, j in sorted(degree_new.items(), key=lambda x: x[1], reverse=True)]
                if len(new_selected_nei) >= gap:  # 满足了数量
                    for i in range(gap):
                        selected_nei.append(new_selected_nei[i])

                    break

                elif len(new_selected_nei) < gap:  # 没满足
                    for new in new_selected_nei:
                        selected_nei.append(new)

                        gap -= 1

                    neis = new_order_list
                    for each in neis:
                        high_nei.add(each)
                count += 1

            for neii in selected_nei:
                if neii not in subset:
                    subset.append(neii)
                    selected_degree.append(len(G.adj[neii]))
                    selected_rep.append(rep_dict[neii])
            padding = L - len(subset)
            node_subgraph, node_A = generate_subgraph(G, subset)
            node_B = transform(node_A, selected_degree, selected_rep)
            if padding == 0:
                data_dict[node] = node_B
            else:
                node_B_padding = np.zeros([L, L])
                for row in range(node_B.shape[0]):
                    node_B_padding[row, :node_B.shape[0]] = node_B[row, :]
                data_dict[node] = node_B_padding
        else:  # 当节点为孤立节点时，直接用一个L*L的零矩阵来表示
            data_dict[node] = np.zeros([L, L])
    return data_dict


def tip(t, s):
    print(t * int(100 - 2 * len(s)), s, t * int(100 - 2 * len(s)))


# 训练模型
def train_model(loader, model, num_epochs, lr, path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()  # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 优化函数
    loss_list = []  # 存放loss的列表
    for epoch in range(num_epochs):
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            labels = torch.squeeze(labels)
            rs = model(data)
            _, predicted = torch.max(rs.data, 1)
            loss = criterion(rs, labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            loss_list.append(loss.data)
        if epoch % 100 == 0:
            print("Loss:{}".format(loss.data))
    if path:
        torch.save(model, path)
    return model, loss_list


def train_model1(loader, model, num_epochs, lr, path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # criterion = nn.CrossEntropyLoss()  # 损失函数
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 优化函数
    loss_list = []  # 存放loss的列表
    for epoch in range(num_epochs):
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            labels = torch.squeeze(labels)
            # print('labels', labels)
            # print('data', data)
            rs = model(data)
            # print('rs', rs)
            # _, predicted = torch.max(rs.data, 1)
            onehot_target = torch.eye(2)[labels.long(), :]
            loss = criterion(rs, onehot_target.view(rs.shape))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            loss_list.append(loss.data)
        if epoch % 20 == 0:
            print(epoch, "Loss:{}".format(loss.data))
    if path:
        torch.save(model, path)
    return model, loss_list


def train_model_reg(loader, model, num_epochs, lr, path=None):
    """回归训练模型
        Parameters:
            loader: pytorch dataloader
            num_epochs: 训练的轮数
            lr:学习率
            path:模型存放路径
        return:
            model:训练好的模型
            loss_list:不同轮数的loss变化
        """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()  # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 优化函数
    loss_list = []  # 存放loss的列表
    for epoch in tqdm(range(num_epochs)):
        for data, targets in loader:
            data = data.to(device)
            targets = targets.float().to(device)
            pred = model(data)
            loss = criterion(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            loss_list.append(loss.data)
        if epoch % 50 == 0:
            print(epoch, "Loss:{}".format(loss.data))
    if path:
        torch.save(model, path)
    return model, loss_list


def undersampling(L, df_label, df1, df2, df3):
    pos_sample = df_label[df_label['isHPU'] == 1]
    neg_sample = df_label[df_label['isHPU'] == 0]

    num = min(len(pos_sample), len(neg_sample))

    pos_us = random.sample(list(np.array(pos_sample['user_id'])), num)
    neg_us = random.sample(list(np.array(neg_sample['user_id'])), num)
    pos_ratings = df2[df2.user_id.isin(pos_us)]
    neg_ratings = df2[df2.user_id.isin(neg_us)]
    pos_dict = embeddings(pos_ratings, df1, df3, L)
    neg_dict = embeddings(neg_ratings, df1, df3, L)
    pos_set = torch.empty(len(pos_dict), 1, L, L)
    for inx, matrix in enumerate(pos_dict.values()):
        pos_set[inx, :, :, :] = torch.from_numpy(matrix)
    # print(pos_set)
    pos_label = torch.empty(len(pos_dict), 1)
    for i in range(len(pos_dict)):
        pos_label[i, :] = 1
    # print(pos_label)
    neg_set = torch.empty(len(neg_dict), 1, L, L)
    for inx, matrix in enumerate(neg_dict.values()):
        neg_set[inx, :, :, :] = torch.from_numpy(matrix)
    # print(neg_set)
    neg_label = torch.empty(len(neg_dict), 1)
    for i in range(len(neg_dict)):
        neg_label[i, :] = 0
    # print(neg_label)
    torch_set = torch.cat((pos_set, neg_set), 0)
    torch_label = torch.cat((pos_label, neg_label), 0)

    return torch_set, torch_label


def sampling(L, df_label, df1, df2, df3):
    pos_sample = df_label[df_label['isHPU'] == 1]
    neg_sample = df_label[df_label['isHPU'] == 0]

    pos_num = len(pos_sample)
    neg_num = len(neg_sample)

    mm = min(pos_num, neg_num)
    total_num = max((pos_num + neg_num) // 2, mm)

    if pos_num >= total_num:
        # print('随机取total_num个样本')
        pos_us = random.sample(list(np.array(pos_sample['user_id'])), total_num)
        pos_ratings = df2[df2.user_id.isin(pos_us)]
        pos_dict = embeddings(pos_ratings, df1, df3, L)
        pos_set = torch.empty(len(pos_dict), 1, L, L)
        for inx, matrix in enumerate(pos_dict.values()):
            pos_set[inx, :, :, :] = torch.from_numpy(matrix)
        pos_label = torch.empty(len(pos_dict), 1)
        for i in range(len(pos_dict)):
            pos_label[i, :] = 1
    else:
        pos_ratings = df2[df2.user_id.isin(pos_sample['user_id'])]
        pos_dict = embeddings(pos_ratings, df1, df3, L)
        pos_oversample = data_oversampling(pos_sample, L, total_num, df1, df2, df3)
        pos_dict.update(pos_oversample)
        pos_set = torch.empty(len(pos_dict), 1, L, L)
        for inx, matrix in enumerate(pos_dict.values()):
            pos_set[inx, :, :, :] = torch.from_numpy(matrix)
        pos_label = torch.empty(len(pos_dict), 1)
        for i in range(len(pos_dict)):
            pos_label[i, :] = 1
        # print('随机生成total_num-pos_num')

    if neg_num >= total_num:
        # print('随机取total_num个样本')
        neg_us = random.sample(list(np.array(neg_sample['user_id'])), total_num)
        neg_ratings = df2[df2.user_id.isin(neg_us)]
        neg_dict = embeddings(neg_ratings, df1, df3, L)
        neg_set = torch.empty(len(neg_dict), 1, L, L)
        for inx, matrix in enumerate(neg_dict.values()):
            neg_set[inx, :, :, :] = torch.from_numpy(matrix)
        # print(neg_set)
        neg_label = torch.empty(len(neg_dict), 1)
        for i in range(len(neg_dict)):
            neg_label[i, :] = 0
    else:
        neg_ratings = df2[df2.user_id.isin(neg_sample['user_id'])]
        neg_dict = embeddings(neg_ratings, df1, df3, L)
        neg_oversample = data_oversampling(neg_sample, L, total_num, df1, df2, df3)
        neg_dict.update(neg_oversample)
        neg_set = torch.empty(len(neg_dict), 1, L, L)
        for inx, matrix in enumerate(neg_dict.values()):
            neg_set[inx, :, :, :] = torch.from_numpy(matrix)
        neg_label = torch.empty(len(neg_dict), 1)
        for i in range(len(neg_dict)):
            neg_label[i, :] = 0
        # print('随机生成total_num-neg_num')

    torch_set = torch.cat((pos_set, neg_set), 0)
    torch_label = torch.cat((pos_label, neg_label), 0)

    return torch_set, torch_label


def sampling2(G, L, df_label, df_objects_quality, df_ratings, df_dfa, rep_dict):
    pos_sample = df_label[df_label['isHPU'] == 1]
    neg_sample = df_label[df_label['isHPU'] == 0]

    pos_num = len(pos_sample)
    neg_num = len(neg_sample)

    # 均衡采样机制
    mm = min(pos_num, neg_num)
    total_num = max((pos_num + neg_num) // 2, mm)

    node_rep = embeddings2(G, L, rep_dict)

    if pos_num >= total_num:
        # print('随机取total_num个样本')
        pos_us = random.sample(list(np.array(pos_sample['user_id'])), total_num)
        pos_ratings = df_ratings[df_ratings.user_id.isin(pos_us)]
        pos_dict = embeddings(pos_ratings, df_objects_quality, df_dfa, L)
        # 使用列表推导式找到属于特定键的值
        pos_rep = [node_rep[key] for key in node_rep if key in pos_us]
        pos_set = torch.empty(len(pos_dict), 2, L, L)
        for inx, matrix in enumerate(pos_dict.values()):
            pos_set[inx, :, :, :] = torch.from_numpy(np.stack((matrix, pos_rep[inx])))
        pos_label = torch.empty(len(pos_dict), 1)
        for i in range(len(pos_dict)):
            pos_label[i, :] = 1
    else:
        pos_ratings = df_ratings[df_ratings.user_id.isin(pos_sample['user_id'])]
        pos_dict = embeddings(pos_ratings, df_objects_quality, df_dfa, L)
        pos_oversample = data_oversampling(pos_sample, L, total_num, df_objects_quality, df_ratings, df_dfa)
        pos_dict.update(pos_oversample)
        # 使用列表推导式找到属于特定键的值
        pos_rep = [node_rep[key] for key in node_rep if key in list(pos_sample['user_id'])]
        pos_set = torch.empty(len(pos_dict), 2, L, L)
        for inx, matrix in enumerate(pos_dict.values()):
            pos_set[inx, :, :, :] = torch.from_numpy(
                np.stack((matrix, random.choice(pos_rep) if inx >= len(pos_rep) else pos_rep[inx]))
                )
        pos_label = torch.empty(len(pos_dict), 1)
        for i in range(len(pos_dict)):
            pos_label[i, :] = 1
        # print('随机生成total_num-pos_num')

    if neg_num >= total_num:
        # print('随机取total_num个样本')
        neg_us = random.sample(list(np.array(neg_sample['user_id'])), total_num)
        neg_ratings = df_ratings[df_ratings.user_id.isin(neg_us)]
        neg_dict = embeddings(neg_ratings, df_objects_quality, df_dfa, L)
        # 使用列表推导式找到属于特定键的值
        neg_rep = [node_rep[key] for key in node_rep if key in neg_us]
        neg_set = torch.empty(len(neg_dict), 2, L, L)
        for inx, matrix in enumerate(neg_dict.values()):
            neg_set[inx, :, :, :] = torch.from_numpy(np.stack((matrix, neg_rep[inx])))
        # print(neg_set)
        neg_label = torch.empty(len(neg_dict), 1)
        for i in range(len(neg_dict)):
            neg_label[i, :] = 0
    else:
        neg_ratings = df_ratings[df_ratings.user_id.isin(neg_sample['user_id'])]
        neg_dict = embeddings(neg_ratings, df_objects_quality, df_dfa, L)
        neg_oversample = data_oversampling(neg_sample, L, total_num, df_objects_quality, df_ratings, df_dfa)
        neg_dict.update(neg_oversample)
        # 使用列表推导式找到属于特定键的值
        neg_rep = [node_rep[key] for key in node_rep if key in list(neg_sample['user_id'])]
        neg_set = torch.empty(len(neg_dict), 2, L, L)
        for inx, matrix in enumerate(neg_dict.values()):
            neg_set[inx, :, :, :] = torch.from_numpy(
                np.stack((matrix, random.choice(list(neg_rep.values())) if inx >= len(neg_rep) else neg_rep[inx]))
                )
        neg_label = torch.empty(len(neg_dict), 1)
        for i in range(len(neg_dict)):
            neg_label[i, :] = 0
        # print('随机生成total_num-neg_num')

    torch_set = torch.cat((pos_set, neg_set), 0)
    torch_label = torch.cat((pos_label, neg_label), 0)

    return torch_set, torch_label


def nosampling(L, df_label, df1, df2, df3):
    pos_sample = df_label[df_label['isHPU'] == 1]
    neg_sample = df_label[df_label['isHPU'] == 0]

    # print('随机取total_num个样本')
    pos_ratings = df2[df2.user_id.isin(set(pos_sample['user_id']))]
    pos_dict = embeddings(pos_ratings, df1, df3, L)
    pos_set = torch.empty(len(pos_dict), 1, L, L)
    for inx, matrix in enumerate(pos_dict.values()):
        pos_set[inx, :, :, :] = torch.from_numpy(matrix)
    pos_label = torch.empty(len(pos_dict), 1)
    for i in range(len(pos_dict)):
        pos_label[i, :] = 1

    neg_ratings = df2[df2.user_id.isin(set(neg_sample['user_id']))]
    neg_dict = embeddings(neg_ratings, df1, df3, L)
    neg_set = torch.empty(len(neg_dict), 1, L, L)
    for inx, matrix in enumerate(neg_dict.values()):
        neg_set[inx, :, :, :] = torch.from_numpy(matrix)
    # print(neg_set)
    neg_label = torch.empty(len(neg_dict), 1)
    for i in range(len(neg_dict)):
        neg_label[i, :] = 0

    torch_set = torch.cat((pos_set, neg_set), 0)
    torch_label = torch.cat((pos_label, neg_label), 0)

    return torch_set, torch_label


def oversampling(L, df_label, df1, df2, df3):
    pos_sample = df_label[df_label['isHPU'] == 1]
    neg_sample = df_label[df_label['isHPU'] == 0]

    pos_num = len(pos_sample)
    neg_num = len(neg_sample)

    total_num = max(len(pos_sample), len(neg_sample))

    if pos_num >= total_num:
        # print('随机取total_num个样本')
        pos_us = random.sample(list(np.array(pos_sample['user_id'])), total_num)
        pos_ratings = df2[df2.user_id.isin(pos_us)]
        pos_dict = embeddings(pos_ratings, df1, df3, L)
        pos_set = torch.empty(len(pos_dict), 1, L, L)
        for inx, matrix in enumerate(pos_dict.values()):
            pos_set[inx, :, :, :] = torch.from_numpy(matrix)
        pos_label = torch.empty(len(pos_dict), 1)
        for i in range(len(pos_dict)):
            pos_label[i, :] = 1
    else:
        pos_ratings = df2[df2.user_id.isin(pos_sample['user_id'])]
        pos_dict = embeddings(pos_ratings, df1, df3, L)
        pos_oversample = data_oversampling(pos_sample, L, total_num, df1, df2, df3)
        pos_dict.update(pos_oversample)
        pos_set = torch.empty(len(pos_dict), 1, L, L)
        for inx, matrix in enumerate(pos_dict.values()):
            pos_set[inx, :, :, :] = torch.from_numpy(matrix)
        pos_label = torch.empty(len(pos_dict), 1)
        for i in range(len(pos_dict)):
            pos_label[i, :] = 1
        # print('随机生成total_num-pos_num')

    if neg_num >= total_num:
        # print('随机取total_num个样本')
        neg_us = random.sample(list(np.array(neg_sample['user_id'])), total_num)
        neg_ratings = df2[df2.user_id.isin(neg_us)]
        neg_dict = embeddings(neg_ratings, df1, df3, L)
        neg_set = torch.empty(len(neg_dict), 1, L, L)
        for inx, matrix in enumerate(neg_dict.values()):
            neg_set[inx, :, :, :] = torch.from_numpy(matrix)
        # print(neg_set)
        neg_label = torch.empty(len(neg_dict), 1)
        for i in range(len(neg_dict)):
            neg_label[i, :] = 0
    else:
        neg_ratings = df2[df2.user_id.isin(neg_sample['user_id'])]
        neg_dict = embeddings(neg_ratings, df1, df3, L)
        neg_oversample = data_oversampling(neg_sample, L, total_num, df1, df2, df3)
        neg_dict.update(neg_oversample)
        neg_set = torch.empty(len(neg_dict), 1, L, L)
        for inx, matrix in enumerate(neg_dict.values()):
            neg_set[inx, :, :, :] = torch.from_numpy(matrix)
        neg_label = torch.empty(len(neg_dict), 1)
        for i in range(len(neg_dict)):
            neg_label[i, :] = 0
        # print('随机生成total_num-neg_num')

    torch_set = torch.cat((pos_set, neg_set), 0)
    torch_label = torch.cat((pos_label, neg_label), 0)

    return torch_set, torch_label


def data_oversampling(sample, L, num, df1, df2, df3):
    tip('*', 'Starting data oversampling ...')
    data_dict1 = {}
    sample_u = list(np.array(sample['user_id']))
    fix_L = L
    L = min(L, len(sample)+1)
    for i in tqdm(range(num - len(sample))):
        us = random.sample(sample_u, L - 1)
        um = list()
        ur = list()
        udfa = list()
        for j in range(L - 1):
            tmp_df = df2[df2['user_id'] == us[j]].copy(deep=True)
            tmp_df['time'] = tmp_df.apply(lambda a: -a['timestamp'], axis=1)
            tmp_df = tmp_df.sort_values(by=['rating', 'time'], axis=0, ascending=False)
            movies_u = np.array(tmp_df['movie_id'].head(L - 1))
            if len(movies_u) <= j:
                um.append(movies_u[-1])
                ur.append(float(df2[(df2['movie_id'] == movies_u[-1]) & (df2['user_id'] == us[j])]['rating']))
            else:
                um.append(movies_u[j])
                ur.append(float(df2[(df2['movie_id'] == movies_u[j]) & (df2['user_id'] == us[j])]['rating']))
            udfa.append(float(df3[df3['user_id'] == us[j]]['dfa']))
        qty_m = list()  # RBeta电影的质量
        for m in um:
            qty_m.append(float(df1[df1['movie_id'] == m]['quality']))
        A = np.zeros([fix_L, fix_L])
        udfa = pd.DataFrame(udfa, columns=['dfa'])
        A[0, 0] = udfa['dfa'].mean()  # 用户DFA值
        A[0, 1:L] = ur
        A[1:L, 0] = ur
        for k in range(1, L):
            A[k, k] = qty_m[k - 1]
        for r in range(1, L):
            for s in range(r + 1, L):
                A[r, s] = A[s, r] = 1 / math.exp(abs(abs(ur[r - 1] - ur[s - 1]) - abs(qty_m[r - 1] - qty_m[s - 1])))
        data_dict1[i] = A
    tip('*', 'Data oversampling over!!')

    return data_dict1
