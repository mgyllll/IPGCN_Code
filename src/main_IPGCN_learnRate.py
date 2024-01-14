#!/usr/bin/env python
# coding: utf-8

"""
Python    : 3.7
@File     : main_IPGCN_learnRate.py
@Copyright: MGYL
@Author   : LuoYong
@Date     : 2022-09-07
@Desc     : NULL
"""

import os
import torch
import warnings
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
import my_package.my_utils as Utils
import my_package.my_models as Models
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    """
    输入：
        ratings_u50.csv——评分网络（用户至少50评分）
        users_reputation.csv——计算的用户声誉
        objects_quality.csv——计算的商品（电影或图书）质量
        users_dfa.csv——计算的用户评分序列相关性dfa值
        users_perceptibility.csv——计算的用户感知力值
    超参数：
        L = 28 ——特征矩阵大小
        batch_size = 64 ——训练批次
        num_epochs = range(5, 205, 5) ——训练轮次
        lr = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001] ——使用固定fix学习率
        q = range(5,55,5) ——根据感知力值划分的标签比例，前q%为感知用户，剩余为其他用户
    输出：
        不同划分比例q下，随训练轮次增加的各项评价指标性能（准确值、精确值、召回值、F1值、AUC值）
    """

    # 数据集类型
    data_type = 'netflix'
    # 数据文件夹
    data_path = '../data/moviesData/' + data_type

    exp_file = data_path + '/gcn_exp20231224'

    # 使用os.makedirs()创建文件夹，如果路径不存在则递归创建
    if not os.path.exists(exp_file):
        os.makedirs(exp_file)

    # 用户度值degree
    ratings_u50 = pd.read_csv(data_path + '/ratings_u50.csv')
    users_degree = ratings_u50.groupby('user_id').size().reset_index(name='degree')

    # 用户声誉reputation
    users_reputation = pd.read_csv(data_path + '/users_reputation.csv')
    objects_quality = pd.read_csv(data_path + '/objects_quality.csv')

    # 用户DFA
    users_dfa = pd.read_csv(data_path + '/users_dfa.csv')

    # 特征合并
    df1 = pd.merge(users_degree, users_reputation, how='outer', on='user_id')
    df2 = pd.merge(df1, users_dfa, how='outer', on='user_id')

    # 用户感知力
    users_perceptibility = pd.read_csv(data_path + '/users_perceptibility.csv')

    # ##################################################卷积神经网络训练#####################################################
    # 参数设置
    seed = 100
    torch.manual_seed(seed)
    L = 28
    batch_size = 64
    # num_epochs = 100
    # lr = 0.0001  # 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5
    # 设置 k 折交叉验证
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

    for q in range(5, 55, 5):
        print('parameter q:', q, '%')
        # 构建感知用户标签
        users_P1 = users_perceptibility.head(int(len(users_perceptibility) * (q / 100))).copy()
        users_P1['isHPU'] = 1
        users_P2 = users_perceptibility.tail(
            len(users_perceptibility) - int(len(users_perceptibility) * (q / 100))).copy()
        users_P2['isHPU'] = 0
        users_HPU = pd.concat([users_P1, users_P2])
        users_HPU.drop("d", axis=1, inplace=True)
        users_HPU.drop("D", axis=1, inplace=True)
        users_HPU.drop("perceptibility", axis=1, inplace=True)
        df = pd.merge(df2, users_HPU, how='right', on='user_id')
        df = df.sort_values(by='user_id', axis=0, ascending=True)

        sampleX, labelY = Utils.oversampling(L, df, objects_quality, ratings_u50, users_dfa)

        for lr in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
            print('学习率为', lr)
            PCNNList = list()
            for ep in range(5, 120, 10):
                pcnnList = list()
                print('parameter epoch:', ep)
                for train_idx, val_idx in kfold.split(sampleX, labelY):
                    xtrain1, xval1 = sampleX[train_idx], sampleX[val_idx]
                    ytrain1, yval1 = labelY[train_idx], labelY[val_idx]

                    # 创建 DataLoader
                    train_data = TensorDataset(xtrain1, ytrain1)
                    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
                    rcnn = Models.CNN1(L)
                    rcnn, loss = Utils.train_model1(train_loader, rcnn, ep, lr, path=None)

                    for j in range(50):
                        _, predicted2 = torch.max(rcnn(xval1), 1)
                        labels2 = torch.squeeze(yval1).long()
                        fpr, tpr, threshold = roc_curve(labels2, predicted2)  # 计算真正率和假正率
                        roc_auc = auc(fpr, tpr)  # 计算auc的值
                        pcnnList.append([accuracy_score(labels2, predicted2), precision_score(labels2, predicted2),
                                         recall_score(labels2, predicted2),
                                         f1_score(labels2, predicted2), roc_auc])
                        print('pcnn>>>>', accuracy_score(labels2, predicted2), precision_score(labels2, predicted2),
                              recall_score(labels2, predicted2),
                              f1_score(labels2, predicted2), roc_auc)

                pcnnList = pd.DataFrame(pcnnList, columns=['accuracy', 'precision', 'recall', 'f1', 'auc'])
                PCNNList.append(
                    [ep, pcnnList['accuracy'].mean(), pcnnList['precision'].mean(), pcnnList['recall'].mean(),
                     pcnnList['f1'].mean(),
                     pcnnList['auc'].mean()])
            PCNNList = pd.DataFrame(PCNNList, columns=['ep', 'accuracy', 'precision', 'recall', 'f1', 'auc'])
            PCNNList.to_csv(exp_file + '/rs20231229_q' + str(q) + '_lr' + str(lr).split('.')[1] + '.csv', index=False,
                            encoding="utf_8_sig")
