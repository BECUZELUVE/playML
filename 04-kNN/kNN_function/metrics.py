# -*- coding:utf-8 -*-
# Author: Icey
# Data: 2021/1/9 16:14

def accuracy_score(y_true, y_predict):
    '''计算y_true和y_predict之间的准确率'''
    assert y_true.shape[0] == y_predict.shape[0], \
        "the size of y_true must be equal to the size of y_predict"

    return sum(y_true == y_predict) / len(y_true)
