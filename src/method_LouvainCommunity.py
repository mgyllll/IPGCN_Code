#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/11/18 14:06
# @Author : Luo Yong(MGYL)
# @File : method_LouvainCommunity.py
# @Software: PyCharm
import networkx as nx
import community
from tqdm import tqdm
import pandas as pd


def best_louvain_partition(G, num_runs=10):
    best_modularity = -float('inf')
    best_partition = None

    for i in tqdm(range(num_runs)):
        partition = community.best_partition(G, weight='weight')
        modularity = community.modularity(partition, G, weight='weight')
        # print(i, " round: ", "modularity=", modularity)
        if modularity > best_modularity:
            best_modularity = modularity
            best_partition = partition

    return best_partition, best_modularity


if __name__ == "__main__":
    # 设定对应数据集
    # ../data/moviesData/netflix/
    # ../data/moviesData/ml_1m/
    # ../data/moviesData/douban/
    # ../data/booksData/amazon/
    path = "../data/moviesData/netflix/"

    # 用户-对象评分数据集加载
    objects_network_u50_filter = pd.read_csv(path + 'objects_network_u50_filter.csv')

    G = nx.Graph()
    for idx, row in tqdm(objects_network_u50_filter.iterrows(), total=len(objects_network_u50_filter)):
        # G.add_edge(row['i'], row['j'], weight=row['relation'])
        G.add_edge(row['i'], row['j'])

    partition, modularity = best_louvain_partition(G, num_runs=10)
    objects_community = pd.DataFrame({
        "id": partition.keys(),
        "comm": partition.values()
    })
    objects_community.to_csv(path + 'objects_network_u50_community.csv', index=False, encoding="utf_8_sig")
    print("Best partition:", len(set(partition.values())))
    print("Best modularity:", modularity)
