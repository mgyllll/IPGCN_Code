#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/11/6 8:56
# @Author : Luo Yong(MGYL)
# @File : generateNet_Feature_labels.py
# @Software: PyCharm

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

path = "../data/moviesData/douban/"

ratings_u50 = pd.read_csv(path + "ratings_u50.csv")
users_degree = ratings_u50.groupby('user_id').size().reset_index(name='degree')
# 创建MinMaxScaler对象
scaler = MinMaxScaler()
# 使用fit_transform来标准化列
users_degree['degree_normalized'] = scaler.fit_transform(users_degree[['degree']])
users_degree.drop("degree", axis=1, inplace=True)
users_dfa = pd.read_csv(path + "users_dfa.csv")
users_reputation = pd.read_csv(path + "users_reputation.csv")
# 三个特征值合并为一张表
df1 = pd.merge(users_degree, users_reputation, how='outer', on='user_id')
df2 = pd.merge(df1, users_dfa, how='outer', on='user_id')
# 感知用户界定点划分
users_perceptibility = pd.read_csv(path + "users_perceptibility.csv")
users_perceptibility = users_perceptibility.sort_values(by='perceptibility', ascending=False)
for q in range(5, 55, 5):
    users_P1 = users_perceptibility.head(int(len(users_perceptibility) * (q / 100))).copy()
    users_P1['isHPU'] = 1
    users_P2 = users_perceptibility.tail(len(users_perceptibility) - int(len(users_perceptibility) * (q / 100))).copy()
    users_P2['isHPU'] = 0
    users_HPU = pd.concat([users_P1, users_P2])
    users_HPU.drop("d", axis=1, inplace=True)
    users_HPU.drop("D", axis=1, inplace=True)
    users_HPU.drop("perceptibility", axis=1, inplace=True)
    idx_feature_label = pd.merge(df2, users_HPU, how='right', on='user_id')
    idx_feature_label = idx_feature_label.reset_index()
    idx_feature_label = idx_feature_label.rename(columns={'index': 'id'})  # 将索引列重命名为 "id"
    idx_feature_label.to_csv(path + "idx_feature_label_" + str(q) + ".csv", index=False, encoding="utf_8_sig")

users_network_u50_filter_weight = pd.read_csv(path + "users_network_u50_filter.csv")
# 将DataFrame保存为以制表符分隔的文件
net = users_network_u50_filter_weight[['i', 'j']].copy()
# 将列转换为整数类型
# net['i'] = net['i'].astype(int)
# net['j'] = net['j'].astype(int)
net = pd.merge(net, idx_feature_label[['user_id', 'id']], left_on='i', right_on='user_id', how='left')
net = pd.merge(net, idx_feature_label[['user_id', 'id']], left_on='j', right_on='user_id', how='left')

net.to_csv(path + "net.csv", index=False, encoding="utf_8_sig")




