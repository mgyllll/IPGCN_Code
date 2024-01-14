#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/12/16 11:12
# @Author : Luo Yong(MGYL)
# @File : baseline_TraditionalMachineLearning.py
# @Software: PyCharm
# @Desc     : NULL
import os
import warnings
import pandas as pd
from sklearn import svm
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    '''
    传统机器学习方法：支持向量机、随机森林、梯度提升
    评价指标：精确值、召回值、F1值、AUC值
    '''
    # 初始输入为 ratings_u50.csv movies_label.csv
    data_type = 'amazon'
    # 数据文件夹
    data_path = '../data/booksData/' + data_type
    # 实验日期
    exp_date = '20231225'
    # 实验文件夹
    file_path = data_path + '/ml_exp' + exp_date
    # 使用os.makedirs()创建文件夹，如果路径不存在则递归创建
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # 用户度值degree
    ratings_u50 = pd.read_csv(data_path + '/ratings_u50.csv')
    users_degree = ratings_u50.groupby('user_id').size().reset_index(name='degree')

    # 用户声誉reputation
    users_reputation = pd.read_csv(data_path + '/users_reputation.csv')
    objects_quality = pd.read_csv(data_path + '/objects_quality.csv')

    # 用户DFA
    users_dfa = pd.read_csv(data_path + '/users_dfa.csv')

    df1 = pd.merge(users_degree, users_reputation, how='outer', on='user_id')
    df2 = pd.merge(df1, users_dfa, how='outer', on='user_id')

    # 用户感知力
    users_perceptibility = pd.read_csv(data_path + '/users_perceptibility.csv')

    # ##################################################卷积神经网络训练#####################################################
    # 参数设置
    seed = 100

    GBMList = list()
    SVMList = list()
    RFList = list()
    DTList = list()
    KNNList = list()
    NBList = list()
    LRList = list()
    MLPList = list()
    for q in range(5, 55, 5):
        print('parameter q:', q, '%')
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

        y = df.isHPU
        x = df.drop(['user_id', 'isHPU'], axis=1)

        # 使用 RandomUnderSampler 进行欠抽样
        # ratio 参数表示正样本与负样本的比例，'auto' 表示自动计算
        rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = rus.fit_resample(x, y)

        # scaler = MinMaxScaler()
        # x = pd.DataFrame(scaler.fit_transform(df[['reputation', 'dfa', 'degree']]),
        #                  columns=['reputation', 'dfa', 'degree'])
        # xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=seed)
        # 设置 k 折交叉验证
        kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

        gbmList = list()
        svmList = list()
        rfList = list()
        dtList = list()
        knnList = list()
        lrList = list()
        mlpList = list()
        for train_idx, val_idx in kfold.split(X_resampled, y_resampled):
            xtrain, xtest = X_resampled.iloc[train_idx], X_resampled.iloc[val_idx]
            ytrain, ytest = y_resampled.iloc[train_idx], y_resampled.iloc[val_idx]

            gbc = GradientBoostingClassifier().fit(xtrain, ytrain)
            rfc = RandomForestClassifier().fit(xtrain, ytrain)
            svmc = svm.SVC().fit(xtrain, ytrain)
            model_dt = DecisionTreeClassifier().fit(xtrain, ytrain)
            model_knn = KNeighborsClassifier().fit(xtrain, ytrain)
            model_lr = LogisticRegression().fit(xtrain, ytrain)
            model_mlp = MLPClassifier().fit(xtrain, ytrain)

            for j in range(10):
                pre_gbc = gbc.predict(xtest)
                gbmList.append(
                    [accuracy_score(ytest, pre_gbc), precision_score(ytest, pre_gbc), recall_score(ytest, pre_gbc),
                     f1_score(ytest, pre_gbc),
                     roc_auc_score(ytest, pre_gbc)])

                pre_rfc = rfc.predict(xtest)
                rfList.append(
                    [accuracy_score(ytest, pre_rfc), precision_score(ytest, pre_rfc), recall_score(ytest, pre_rfc),
                     f1_score(ytest, pre_rfc),
                     roc_auc_score(ytest, pre_rfc)])

                pre_svmc = svmc.predict(xtest)
                svmList.append(
                    [accuracy_score(ytest, pre_svmc), precision_score(ytest, pre_svmc), recall_score(ytest, pre_svmc),
                     f1_score(ytest, pre_svmc),
                     roc_auc_score(ytest, pre_svmc)])

                pre_dt = model_dt.predict(xtest)
                dtList.append(
                    [accuracy_score(ytest, pre_dt), precision_score(ytest, pre_dt), recall_score(ytest, pre_dt),
                     f1_score(ytest, pre_dt),
                     roc_auc_score(ytest, pre_dt)])

                pre_knn = model_knn.predict(xtest)
                knnList.append(
                    [accuracy_score(ytest, pre_knn), precision_score(ytest, pre_knn), recall_score(ytest, pre_knn),
                     f1_score(ytest, pre_knn),
                     roc_auc_score(ytest, pre_knn)])

                pre_lr = model_lr.predict(xtest)
                lrList.append(
                    [accuracy_score(ytest, pre_lr), precision_score(ytest, pre_lr), recall_score(ytest, pre_lr),
                     f1_score(ytest, pre_lr),
                     roc_auc_score(ytest, pre_lr)])

                pre_mlp = model_mlp.predict(xtest)
                mlpList.append(
                    [accuracy_score(ytest, pre_mlp), precision_score(ytest, pre_mlp), recall_score(ytest, pre_mlp),
                     f1_score(ytest, pre_mlp),
                     roc_auc_score(ytest, pre_mlp)])

        gbmList = pd.DataFrame(gbmList, columns=['accuracy', 'precision', 'recall', 'f1', 'auc'])
        rfList = pd.DataFrame(rfList, columns=['accuracy', 'precision', 'recall', 'f1', 'auc'])
        svmList = pd.DataFrame(svmList, columns=['accuracy', 'precision', 'recall', 'f1', 'auc'])
        dtList = pd.DataFrame(dtList, columns=['accuracy', 'precision', 'recall', 'f1', 'auc'])
        knnList = pd.DataFrame(knnList, columns=['accuracy', 'precision', 'recall', 'f1', 'auc'])
        lrList = pd.DataFrame(lrList, columns=['accuracy', 'precision', 'recall', 'f1', 'auc'])
        mlpList = pd.DataFrame(mlpList, columns=['accuracy', 'precision', 'recall', 'f1', 'auc'])

        GBMList.append(
            [q, gbmList['accuracy'].mean(), gbmList['precision'].mean(), gbmList['recall'].mean(), gbmList['f1'].mean(),
             gbmList['auc'].mean()])
        RFList.append(
            [q, rfList['accuracy'].mean(), rfList['precision'].mean(), rfList['recall'].mean(), rfList['f1'].mean(),
             rfList['auc'].mean()])
        SVMList.append(
            [q, svmList['accuracy'].mean(), svmList['precision'].mean(), svmList['recall'].mean(), svmList['f1'].mean(),
             svmList['auc'].mean()])
        DTList.append(
            [q, dtList['accuracy'].mean(), dtList['precision'].mean(), dtList['recall'].mean(), dtList['f1'].mean(),
             dtList['auc'].mean()])
        KNNList.append(
            [q, knnList['accuracy'].mean(), knnList['precision'].mean(), knnList['recall'].mean(), knnList['f1'].mean(),
             knnList['auc'].mean()])
        LRList.append(
            [q, lrList['accuracy'].mean(), lrList['precision'].mean(), lrList['recall'].mean(), lrList['f1'].mean(),
             lrList['auc'].mean()])
        MLPList.append(
            [q, mlpList['accuracy'].mean(), mlpList['precision'].mean(), mlpList['recall'].mean(), mlpList['f1'].mean(),
             mlpList['auc'].mean()])

    GBMList = pd.DataFrame(GBMList, columns=['q', 'accuracy', 'precision', 'recall', 'f1', 'auc'])
    RFList = pd.DataFrame(RFList, columns=['q', 'accuracy', 'precision', 'recall', 'f1', 'auc'])
    SVMList = pd.DataFrame(SVMList, columns=['q', 'accuracy', 'precision', 'recall', 'f1', 'auc'])
    DTList = pd.DataFrame(DTList, columns=['q', 'accuracy', 'precision', 'recall', 'f1', 'auc'])
    KNNList = pd.DataFrame(KNNList, columns=['q', 'accuracy', 'precision', 'recall', 'f1', 'auc'])
    LRList = pd.DataFrame(LRList, columns=['q', 'accuracy', 'precision', 'recall', 'f1', 'auc'])
    MLPList = pd.DataFrame(MLPList, columns=['q', 'accuracy', 'precision', 'recall', 'f1', 'auc'])

    GBMList.to_csv(file_path + '/rs_methodXGB.csv', index=False, encoding="utf_8_sig")
    RFList.to_csv(file_path + '/rs_methodRF.csv', index=False, encoding="utf_8_sig")
    SVMList.to_csv(file_path + '/rs_methodSVM.csv', index=False, encoding="utf_8_sig")
    DTList.to_csv(file_path + '/rs_methodDT.csv', index=False, encoding="utf_8_sig")
    KNNList.to_csv(file_path + '/rs_methodKNN.csv', index=False, encoding="utf_8_sig")
    LRList.to_csv(file_path + '/rs_methodLR.csv', index=False, encoding="utf_8_sig")
    MLPList.to_csv(file_path + '/rs_methodMLP.csv', index=False, encoding="utf_8_sig")
