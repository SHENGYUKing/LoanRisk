# -*- coding:utf-8 -*-

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import dataset as ds
import pandas as pd
import numpy as np
import time


def show_acc(a, b, tip):
    acc = a.ravel() == b.ravel()
    print(tip + "Acc： {:.6f}%".format(float(acc.sum()) / a.size * 100))


@ds.time_cost
def train():
    x, y = ds.load_data()

    print('Transforming Dataset ...')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2233)
    # 设置参数
    param = {
        'booster': 'gbtree',
        'max_depth': 5,
        'eta': 0.2,
        'gamma': 0.1,
        'objective': 'multi:softmax',
        'num_class': 2,
        'seed': 2233
    }
    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    print('Done.')
    print('*' * 20)

    print('Training Model ...')
    n_round = 10
    watchlist = [(data_train, 'train'), (data_test, 'eval')]
    bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist)
    y_hat = bst.predict(data_test)
    show_acc(y_hat, y_test, 'XGBoost ')
    print('Done.')
    print('*' * 20)

    print('Saving Model File ...')
    model_name = 'model_' + time.strftime('%Y%m%d_%H%M%S', time.localtime())
    bst.save_model('./model/' + model_name + '.model')
    print('Done.')


train()
