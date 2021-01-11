# -*- coding:utf-8 -*-
# Author: Icey
# Data: 2021/1/9 11:57

import numpy as np
from math import sqrt
from collections import Counter


def kNN_classify(k, X_train, y_train, x):
    '''添加断言，限定k的范围，数据集x和y的维度一致，数据集x和待测数据x的维度一致'''
    assert 1 <= k <= X_train.shape[0], "k must be valid"
    assert X_train.shape[0] == y_train.shape[0], \
        "the size of X_train must equal to the size of y_train"
    assert X_train.shape[1] == x.shape[0], \
        "the feature number of x must equal to X_train"

    distances = [sqrt(np.sum(x_train - x) ** 2) for x_train in X_train]
    nearest = np.argsort(distances)

    topK_y = [y_train[i] for i in nearest[:k]]
    votes = Counter(topK_y)

    return votes.most_common(1)[0][0]
