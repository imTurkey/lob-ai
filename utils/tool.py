import numpy as np

def load_fi_2010(dataset_norm):
    # dataset_norm = "DecPre"
    data = np.loadtxt(f'../data/Train_Dst_NoAuction_{dataset_norm}_CF_7.txt')
    train = data[:, :int(data.shape[1] * 0.8)]
    val = data[:, int(data.shape[1] * 0.8):]

    test1 = np.loadtxt(f'../data/Test_Dst_NoAuction_{dataset_norm}_CF_7.txt')
    test2 = np.loadtxt(f'../data/Test_Dst_NoAuction_{dataset_norm}_CF_8.txt')
    test3 = np.loadtxt(f'../data/Test_Dst_NoAuction_{dataset_norm}_CF_9.txt')
    test = np.hstack((test1, test2, test3))

    return train, val, test

def load_my(symbol, dataset_norm, dataset_path="../data_my"):
    # dataset_norm = "zscore"
    data = np.loadtxt(dataset_path + f'/{symbol}/{dataset_norm}.txt').T
    train = data[:, :int(data.shape[1] * 0.7)]
    test = data[:, int(data.shape[1] * 0.7):]

    val = train[:, int(train.shape[1] * 0.8):]
    train = train[:, :int(train.shape[1] * 0.8)]

    return train, val, test
