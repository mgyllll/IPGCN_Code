#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/11/3 15:25
# @Author : Luo Yong(MGYL)
# @File : project_map_users.py
# @Software: PyCharm
# 导入相关包
import pandas as pd
from tqdm import tqdm
from itertools import combinations
import os

user_dict = {}


def process_combinations(df, path, batch_size):
    global user_dict
    user_dict = {}
    comb_gen = combinations(set(df['user_id']), 2)
    batch = []
    ii = 0
    for i, c in enumerate(comb_gen):
        batch.append(c)
        if i % batch_size == 0 and i != 0:
            # 每次生成指定数量的组合后进行批处理
            if ii > -1:
                batch_process(df, path, batch, ii)
            ii += 1
            batch = []
    # 处理最后一个批次
    if len(batch) > 0:
        batch_process(df, path, batch, ii)


# 执行批处理操作
def batch_process(df, path, batch, ii):
    global user_dict
    print('users_network_u50_' + str(ii) + '.csv')
    common_review_u = list()
    for i, j in tqdm(batch):
        if user_dict.get(i) is not None:
            u1 = user_dict[i]
        else:
            u1 = set(df.loc[df['user_id'] == i, 'movie_id'])
            user_dict[i] = u1
        if user_dict.get(j) is not None:
            u2 = user_dict[j]
        else:
            u2 = set(df.loc[df['user_id'] == j, 'movie_id'])
            user_dict[j] = u2
        uu = u1 & u2
        if uu:
            common_review_u.append([i, j, len(u1), len(u2), len(uu), len(uu) / (len(u1) + len(u2) - len(uu))])
    users_net = pd.DataFrame(common_review_u, columns=['i', 'j', 'ui', 'uj', 'uij', 'relation'])
    users_net.to_csv(path + '/users_network_u50_' + str(ii) + '.csv', index=False, encoding="utf_8_sig")


def users_network_mapping_projection(df, path):
    movie_dict = {}
    user_dict = {}
    users_net = pd.read_csv(path)
    save_path = path.replace('.csv', '_weight.csv')
    if os.path.isfile(save_path):
        users_network_weight = pd.read_csv(save_path)
        print(save_path, '已存在！！！')
    else:
        users_network_weight = list()
        for idx, row in tqdm(users_net.iterrows(), total=len(users_net)):
            if user_dict.get(row['i']) is not None:
                u1 = user_dict[row['i']]
            else:
                u1 = set(df.loc[df['user_id'] == row['i'], 'movie_id'])
                user_dict[row['i']] = u1
            if user_dict.get(row['j']) is not None:
                u2 = user_dict[row['j']]
            else:
                u2 = set(df.loc[df['user_id'] == row['j'], 'movie_id'])
                user_dict[row['j']] = u2
            tmp = 0
            ou = 0  # 欧式距离
            for l in u1 & u2:
                if movie_dict.get(l) is not None:
                    m = movie_dict[l]
                else:
                    m = len(set(df[df['movie_id'] == l]['user_id']))
                    movie_dict[l] = m
                tmp += 1 / m
                # r_il = list(df[(df['movie_id'] == l) & (df['user_id'] == row['i'])]['rating'])[0]
                # r_jl = list(df[(df['movie_id'] == l) & (df['user_id'] == row['j'])]['rating'])[0]

                # ou += pow(r_il-r_jl, 2)
            users_network_weight.append([row['i'], row['j'], row['relation'] * tmp])
        users_network_weight = pd.DataFrame(users_network_weight, columns=['i', 'j', 'wij'])
        users_network_weight['i'] = users_network_weight['i']
        users_network_weight['j'] = users_network_weight['j']
        users_network_weight.to_csv(save_path, index=False, encoding="utf_8_sig")

    return users_network_weight


def first_stop(df1, df2):
    max_q = 0
    stop = 0
    tmp_lst = []
    for q in tqdm(range(100, -1, -1)):
        df = df2[df2['relation'] > q / 100].sort_values('relation')
        p = len(set(df['i']) | set(df['j'])) / len(set(df1['user_id']))
        r = 1 - len(df) / len(df2)
        tmp_lst.append([1 - q / 100, p, r])
        if (p - 1 == 0) and (stop == 0):
            stop = 1
            max_q = q + 1
    tmp_df = pd.DataFrame(tmp_lst, columns=['percent', 'completeness', 'pruning_rate'])

    return df2[df2['relation'] > max_q / 100].sort_values('relation'), tmp_df


if __name__ == "__main__":
    # 设定对应数据集
    # ../data/moviesData/netflix/
    # ../data/moviesData/ml_1m/
    # ../data/moviesData/douban/
    # ../data/booksData/amazon/
    path = "../data/moviesData/douban/"

    # 用户-对象评分数据集加载
    ratings_u50 = pd.read_csv(path + 'ratings_u50.csv')
    print("the num of users:", len(set(ratings_u50['user_id'])))
    print("the num of objects:", len(set(ratings_u50['movie_id'])))
    if not os.path.exists(path + 'users_network_u50_filter.csv'):
        # 二分网络映射为用户单部网络，按批次处理：50000000
        process_combinations(ratings_u50, path, 50000000)

        # 合并映射网络
        no = 0
        fpath1 = path + 'users_network_u50_' + str(no) + '.csv'
        comm = pd.read_csv(fpath1)
        no += 1
        fpath1 = path + '/users_network_u50_' + str(no) + '.csv'
        while os.path.isfile(fpath1):
            print(fpath1, 'yes')
            df = pd.read_csv(fpath1)
            comm = pd.concat([df, comm], axis=0)
            no += 1
            fpath1 = path + '/users_network_u50_' + str(no) + '.csv'

        comm.to_csv(path + '/users_network_u50.csv', index=False, encoding="utf_8_sig")
        # 根据边的绝对强度进行剪枝
        comm_ml_100k_part, pruning_strategy = first_stop(ratings_u50, comm)
        pruning_strategy.to_csv(path + '/users_pruning_strategy.csv', index=False, encoding="utf_8_sig")
        comm_ml_100k_part.to_csv(path + '/users_network_u50_filter.csv', index=False, encoding="utf_8_sig")
