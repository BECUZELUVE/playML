# -*- coding:utf-8 -*-
# Author: Icey
# Data: 2021/1/9 12:52
import numpy as np
from math import sqrt
from collections import Counter
from .metrics import accuracy_score

class KNNClassifier:
    def __init__(self,k):
        '''初始化kNN分类器'''
        assert k>=1,"k must be valid"
        self.k = k
        self._X_train = None  # 变量前加一个下划线代表私有的成员变量，用户不能随意改变
        self._y_train = None

    def fit(self,X_train,y_train):
        '''根据训练数据集来训练分类器'''
        # 因为kNN分类器没有什么算法，所以直接赋值即可
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must equal to the size of y_train"
        assert self.k <= X_train.shape[0], \
            "the feature number of x must equal to X_train"

        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self,X_predict):
        '''给定待预测的数据集X_predict，返回表示X_predict的结果向量'''
        # 先运行fit再运行predict
        assert self._X_train is not None and self._y_train is not None,\
        "must fit before predict!"
        # 预测集中特征个数，要和测试集中的特征个数保持一致
        assert X_predict.shape[1] == self._X_train.shape[1],\
        "the feature number of X_predict must be equal to X_train"

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self,x):
        '''给定单个待测数据x，返回x的预测结果'''
        assert self._X_train.shape[1] == x.shape[0], \
            "the feature number of x must equal to X_train"

        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]
        nearest = np.argsort(distances)
        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)
        return votes.most_common(1)[0][0]

    def score(self,X_test,y_test):
        '''根据测试数据集X_test和y_test确定当前模型的准确度'''
        #  因为是调用的已封装好的函数，这些函数中已经有断言检查的部分了，所以不用再进行断言检查
        y_predict = self.predict(X_test)
        return accuracy_score(y_test,y_predict)

    # 自定义输出实例化对象时的信息
    def __repr__(self):
        return "KNN(k=%d)" % self.k
