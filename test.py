# -*- coding:utf-8 -*-

import xgboost as xgb
import dataset as ds
import pandas as pd
import numpy as np
import time

test_data, id_list = ds.load_test()
model = xgb.Booster()
print('Loading Model ...')
model.load_model(fname='./model/model_20201117_142631.model')
print('Done.')
print('*' * 20)
print('Predicting ...')
y_pred = model.predict(xgb.DMatrix(test_data, nthread=-1))
print('Done.')
print('*' * 20)
print('Generating Answer File ...')
id_list = np.array(id_list).astype(int).reshape(-1, 1)
y_pred = np.array(y_pred).astype(int).reshape(-1, 1)
answer = np.concatenate((id_list, y_pred), axis=1)
answer = pd.DataFrame(answer)
answer.columns = ['id', ds.label]
ans_name = 'ans_' + time.strftime('%Y%m%d_%H%M%S', time.localtime())
print('Saving File ...')
answer.to_csv('./result/' + ans_name + '.csv', sep=',', header=True, index=False)
print('Done.')
