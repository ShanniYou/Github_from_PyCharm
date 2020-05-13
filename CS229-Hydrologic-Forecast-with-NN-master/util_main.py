import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import format_data as fd

def load_data(filename1,filename2):
    """ Load all of the data, including rainfall and runoff """
    rainoff = pd.read_table('rainfall_collins_la.txt', sep='\s', header=None, index_col=None, engine='python')
    runoff = pd.read_table('runoff_collins_la.txt', sep='\s', header=None, index_col=None, engine='python')
    return rainoff,runoff


def feature_collect(rain,run):
    start_date = 20030101
    end_date = 20121231

    '''to construct four features(each lagging for one day)+normalization'''
    X0 = np.array(rain.iloc[0:-3])[:, 1]
    X0 /= np.max(X0)
    X1 = np.array(rain.iloc[1:-2])[:, 1]
    X1 /= np.max(X1)
    Y0 = np.array(run.iloc[6:-3])[:, 1]
    Y0 /= np.max(Y0)
    Y1 = np.array(run.iloc[7:-2])[:, 1]
    Y1 /= np.max(Y1)
    X = np.vstack((X0, X1, Y0, Y1))
    return np.array(X)

def separate_train_valid(filename1,filename2):
    """ Cliping the data into training set and validation set"""

    rain,run  = load_data(filename1,filename2)
    #print(rain,run)
    X = feature_collect(rain,run)

    '''for training set'''
    x_train,x_valid = np.hsplit(X,2)
    y_train = x_train[3,2:]
    y_valid = x_valid[3,2:]

    x_train = np.delete(x_train, [y_train.shape[0],y_train.shape[0]+1], axis=1)
    x_valid = np.delete(x_valid, [y_train.shape[0],y_train.shape[0]+1], axis=1)
    return x_train,x_valid,y_train,y_valid































