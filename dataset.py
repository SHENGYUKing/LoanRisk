# -*- coding:utf-8 -*-

import sklearn.preprocessing as skp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time

FILE_PATH = './data/train.csv'
TEST_PATH = './data/testA.csv'
label = 'isDefault'
rng = np.random.RandomState(1)
m = [
    'Jan',
    'Feb',
    'Mar',
    'Apr',
    'May',
    'Jun',
    'Jul',
    'Aug',
    'Sep',
    'Oct',
    'Nov',
    'Dec']
drop = [
    'id',  # 用户随机编码，无用信息
    'grade',  # 贷款等级，可由贷款子级得知，冗余信息
    'issueDate',  # 贷款发放时间，弱相关信息
    'earliesCreditLine',  # 信用报告最早年份，弱相关信息
    'policyCode',  # 策略代码，仅有唯一值
    'postCode',  # 邮政编码，可用地区编码代替且信息不全，冗余信息
    'applicationType',
    'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13',
    'installment',
    'ficoRangeHigh',
    'pubRec',
    'pubRecBankruptcies'
]
num_fea_std_dict = {
    'loanAmnt': [13764.531844126117, 8229.833179234107],
    'interestRate': [13.117565814391591, 4.485626513250098],
    'installment': [416.05665872775666, 240.01393014758233],
    'employmentTitle': [69828.22155017142, 104265.65461932094],
    'employmentLength': [5.871908993627371, 3.5792858114921042],
    'annualIncome': [69874.42387101255, 37141.55533626742],
    'dti': [17.92615427617815, 8.3685205989137],
    'delinquency_2years': [0.21005529120087593, 0.4926942628937087],
    'ficoRangeLow': [694.9133418345829, 28.09035002383393],
    'ficoRangeHigh': [698.9133418345829, 28.09035002383393],
    'openAcc': [10.387779371072927, 4.100794672071401],
    'pubRec': [0.17802509687951215, 0.4166958083554704],
    'pubRecBankruptcies': [0.123799033106909, 0.32935214058618695],
    'revolBal': [13738.190533842086, 11248.653462696366],
    'revolUtil': [53.279701695453504, 23.885004346671042],
    'totalAcc': [22.20439716382437, 9.568257649023632],
    'title': [525.9512732854022, 2806.4384949171076],
    'n0': [0.337752659948364, 0.776477646488613],
    'n1': [3.2983773198570336, 1.7260141552580641],
    'n2': [5.0793692535935975, 2.3933368509672306],
    'n3': [5.0793692535935975, 2.3933368509672306],
    'n4': [4.1818936616072655, 2.1211244657031467],
    'n5': [7.108668793632994, 3.5937709313703174],
    'n6': [7.530021484675423, 5.615899874308533],
    'n7': [7.320349907462079, 3.253273580312179],
    'n8': [12.772766892098634, 5.9698320597743475],
    'n9': [5.058337757621767, 2.373023693113698],
    'n10': [10.385319962885596, 3.989022339994767],
    'n14': [1.9686892694052598, 1.5044985893218008]
}
num_fea_std_mm_dict = {
    'loanAmnt': [3.187849326286761, -1.6117619343239902],
    'interestRate': [3.2041085326996472, -1.7405742077118576],
    'installment': [3.338070172758736, -1.6680975911755411],
    'employmentTitle': [2.9590067753016127, -0.6697145076690663],
    'employmentLength': [1.153328128510571, -1.6405253178648889],
    'annualIncome': [5.630501314111197, -1.8813004258543433],
    'dti': [3.9557584082566506, -2.2615890171358264],
    'delinquency_2years': [3.6329725016207006, -0.42634003888572203],
    'ficoRangeLow': [3.029035170199849, -2.310876930316126],
    'ficoRangeHigh': [3.029035170199849, -2.310876930316126],
    'openAcc': [4.050975958895274, -2.533113750322406],
    'pubRec': [4.37243395922576, -0.4272303519973122],
    'pubRecBankruptcies': [2.6603773254171434, -0.3758865294956615],
    'revolBal': [5.212606972079556, -1.2213186742219155],
    'revolUtil': [2.939932406348229, -2.230675821621918],
    'totalAcc': [3.7410784856768884, -2.1116067214061776],
    'title': [9.182117752940716, -0.18740880095465515],
    'n0': [4.716487791519885, -0.4349805322480937],
    'n1': [3.3033464197114255, -1.9109792986395748],
    'n2': [3.309450879513847, -2.1222960117548215],
    'n3': [3.309450879513847, -2.1222960117548215],
    'n4': [3.2143829599045013, -1.971545625551483],
    'n5': [3.308872889634287, -1.9780528390334644],
    'n6': [3.645004179823448, -1.3408396967908138],
    'n7': [3.2827396248406258, -2.250148881349115],
    'n8': [3.220732663060564, -1.9720432290592158],
    'n9': [2.9252393317952596, -2.131600191056082],
    'n10': [3.1623488067833065, -2.6034750065850023],
    'n14': [3.344177765472521, -1.30853513813709]
}


def time_cost(func):
    def wrapper(*args, **kw):
        start = time.time()
        re = func(*args, **kw)
        end = time.time()
        print(
            "Function: {}, Cost: {:.3f}sec".format(
                func.__name__,
                (end - start)))
        return re
    return wrapper


def employmentLength_to_value(data):
    if pd.isnull(data):
        return data
    else:
        if data == '10+ years':
            return 10
        elif data == '< 1 year':
            return 0
        else:
            return int(data.split(' ')[0])


def issueDate_to_value(data):
    if pd.isnull(data):
        return data
    else:
        data = pd.to_datetime(str(data), format='%Y-%m-%d')
        start = pd.to_datetime('2000-01-01', format='%Y-%m-%d')
        return (data - start).days


def earliesCreditLine_to_value(data):
    if pd.isnull(data):
        return data
    else:
        year = str(data).split('-')[1]
        month = str(m.index(str(data).split('-')[0]) + 1)
        data = year + '-' + month + '-01'
        data = pd.to_datetime(data, format='%Y-%m-%d')
        start = pd.to_datetime('1900-01-01', format='%Y-%m-%d')
        return (data - start).days


def data_clean(data, fea, sigma=3):
    data_mean = np.mean(data[fea])
    data_std = np.std(data[fea], ddof=1)
    delta = sigma * data_std
    lower_thr = data_mean - delta
    upper_thr = data_mean + delta
    data[fea + '_outlier'] = data[fea].apply(lambda x: str(
        'T') if x > upper_thr or x < lower_thr else str('F'))
    return data


def standardization(data_frame, columns, max_min=False):
    for fea in columns:
        data_mean = np.mean(data_frame[fea])
        data_std = np.std(data_frame[fea])
        # print("'{}': [{}, {}]".format(fea, data_mean, data_std))
        data_frame[fea] = data_frame[fea].apply(
            lambda x: np.divide((x - data_mean), data_std))
        if max_min:
            data_max = np.max(data_frame[fea])
            data_min = np.min(data_frame[fea])
            # print("'{}': [{}, {}]".format(fea, data_max, data_min))
            data_frame[fea] = data_frame[fea].apply(
                lambda x: np.divide((x - data_min), (data_max - data_min)))
    return data_frame


def standardization_test(data_frame, columns, max_min=False):
    for fea in columns:
        data_mean = num_fea_std_dict[fea][0]
        data_std = num_fea_std_dict[fea][1]
        data_frame[fea] = data_frame[fea].apply(
            lambda x: np.divide((x - data_mean), data_std))
        if max_min:
            data_max = num_fea_std_mm_dict[fea][0]
            data_min = num_fea_std_mm_dict[fea][1]
            data_frame[fea] = data_frame[fea].apply(
                lambda x: np.divide((x - data_min), (data_max - data_min)))
    return data_frame


@time_cost
def load_data(file_path=FILE_PATH, save_flag=False):
    print('Loading Dataset ...')
    ori_data = pd.read_csv(file_path)
    print('Done.')
    print('*' * 20)

    ori_data = ori_data.drop(columns=drop)
    cate_fea = [
        'term',
        'subGrade',
        'homeOwnership',
        'verificationStatus',
        'purpose',
        'regionCode',
        'initialListStatus']
    num_fea = list(filter(lambda x: x not in cate_fea, list(ori_data.columns)))
    num_fea.remove(label)

    print('Transforming Datetime Items ...')
    ori_data['employmentLength'] = ori_data['employmentLength'].apply(
        employmentLength_to_value)
    # ori_data['issueDate'] = ori_data['issueDate'].apply(issueDate_to_value)
    # ori_data['earliesCreditLine'] = ori_data['earliesCreditLine'].apply(earliesCreditLine_to_value)
    print('Done.')
    print('*' * 20)

    print('Pre-processing Dataset ...')
    print('-' * 20)
    data = ori_data.copy()
    print('Cleaning Dataset ...')
    for fea in num_fea:
        data = data_clean(data, fea)
        data = data[data[fea + '_outlier'] == 'F']
        data = data.drop(columns=[fea + '_outlier'])
        data = data.reset_index(drop=True)
        # x1 = data[fea + '_outlier'].value_counts()
        # x2 = data.groupby(fea + '_outlier')['isDefault'].sum()
        # print(x1)
        # print(x2)
        # print('- Default Proportion -')
        # print('D/A: {:.2f}%'.format(x2[0] / x1[0] * 100))
        # print('- Outlier Proportion -')
        # if len(x1) < 2:
        #     print('Have No Outlier.')
        # else:
        #     a = x1[1] / x1[0] * 100
        #     b = x2[1] / x2[0] * 100
        #     print(fea + ': {:.2f}%'.format(a))
        #     print('isDefault: {:.2f}%'.format(b))
        #     print('Reject!') if np.abs(a - b) > 5 else print('Accept!')
        # print('-' * 20)
    print('Cleaning Done.')
    print('-' * 20)
    print('Filling NaN Items ...')
    data[num_fea] = data[num_fea].fillna(data[num_fea].median())
    data[cate_fea] = data[cate_fea].fillna(data[cate_fea].mode())
    print('Filling Done.')
    print('-' * 20)
    print('Standardizing Numerical Items ...')
    data = standardization(data, columns=num_fea, max_min=True)
    print('Coding Done.')
    print('-' * 20)

    # print("Printing Heatmap of Feature's Correlation ...")
    # fig = plt.figure()
    # plt.title("Heat Map of some Feature's Correlation")
    # names = num_fea + [label]
    # corr = abs(data[names].corr())
    # ax = sns.heatmap(corr, square=True, vmax=0.8, xticklabels=names, yticklabels=names)
    # plt.savefig('./data/heatmap_fig.png', dpi=400, bbox_inches='tight')
    # print('Printing Done.')
    # print('-' * 20)

    print('One-hot Coding categorized Items ...')
    data = pd.get_dummies(data, columns=cate_fea)
    print('Coding Done.')
    print('-' * 20)
    print('Done.')
    print('*' * 20)

    if save_flag:
        print('Saving Pre-processed Dataset ...')
        data.to_csv('./data/pp_dataset.csv', index=False, sep=',')
        print('Done.')
        print('*' * 20)

    labels = np.asarray(data[label])
    data = data.drop(columns=label)
    dataset = np.asarray(data)
    print('Shape of Dataset: {}'.format(dataset.shape))
    return dataset, labels


@time_cost
def load_test(file_path=TEST_PATH):
    print('Load Test Dataset ...')
    test_data = pd.read_csv(file_path)
    id_list = np.asarray(test_data['id'])
    print('Done.')
    print('*' * 20)

    test_data = test_data.drop(columns=drop)
    cate_fea = [
        'term',
        'subGrade',
        'homeOwnership',
        'verificationStatus',
        'purpose',
        'regionCode',
        'initialListStatus']
    num_fea = list(
        filter(
            lambda x: x not in cate_fea,
            list(
                test_data.columns)))

    print('Transforming Datetime Items ...')
    test_data['employmentLength'] = test_data['employmentLength'].apply(
        employmentLength_to_value)
    print('Done.')
    print('*' * 20)

    print('Pre-processing Dataset ...')
    print('-' * 20)
    data = test_data.copy()
    print('Filling NaN Items ...')
    data[num_fea] = data[num_fea].fillna(data[num_fea].median())
    data[cate_fea] = data[cate_fea].fillna(data[cate_fea].mode())
    print('Filling Done.')
    print('-' * 20)
    print('Standardizing Numerical Items ...')
    data = standardization_test(data, columns=num_fea, max_min=True)
    print('Coding Done.')
    print('-' * 20)
    print('One-hot Coding categorized Items ...')
    data = pd.get_dummies(data, columns=cate_fea)
    print('Coding Done.')
    print('-' * 20)
    print('Done.')
    print('*' * 20)
    testset = np.asarray(data)
    print('Shape of Dataset: {}'.format(testset.shape))
    return testset, id_list
