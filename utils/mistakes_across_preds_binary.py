import numpy as np
import os
from os import walk


"""
Functions
"""


def get_miss_for_n_models(data_arr_misses_train, data_arr_misses_val, n_models):
    """
    :param data_arr_misses_train: Array containing miss score for each train datapoint
    :param data_arr_misses_val:  Array containing miss score for each val datapoint
    """
    # Find datapoints that were miss-classified by n models (BLs)
    miss_hist_train = np.where(data_arr_misses_train >= n_models, 1, 0)
    miss_hist_val = np.where(data_arr_misses_val >= n_models, 1, 0)

    n_miss_train = sum(miss_hist_train)
    n_miss_val = sum(miss_hist_val)

    return n_miss_train, n_miss_val, miss_hist_train, miss_hist_val


# preds_path = "C:/Users/vanya/OneDrive/Desktop/decolearn/decolearn/results/LPBoost_Fin_Preds"
preds_path = "C:/Users/vanya/OneDrive/Desktop/decolearn/decolearn/results/LPBoost_Fin_Preds/Normal_Architecture"
dataset_path = "C:/Users/vanya/OneDrive/Desktop/decolearn/decolearn/soda/dataset/cifar10/binary_data.npy"

""""
Load labels
"""
cifar10 = np.load(dataset_path, allow_pickle=True)
_, y_train_raw, _, y_val_raw = cifar10

# Configure target and non-target
t = 3
nt = 2
y_train = np.where(y_train_raw == t, 1, -1)
y_val = np.where(y_val_raw == t, 1, -1)

""" 
Load predictions
"""
files = list()
# walk(path) returns a generator, gen. instantiates one tuple
dirpath, dirnames, filenames = list(walk(preds_path))[0]

# Extract all train and val preds from .npy-files
all_preds_train = list()
all_preds_val = list()
for file in filenames:
    if file[-4:] != '.npy':
        continue
    loading_path_i = preds_path + '/' + file
    preds_i = np.load(loading_path_i, allow_pickle=True)
    train_i, val_i = preds_i
    all_preds_train.append(train_i)
    all_preds_val.append(val_i)

all_preds_train = np.asarray(all_preds_train)
all_preds_val = np.asarray(all_preds_val)

"""
Get false preds
"""
columns_train = all_preds_train * np.expand_dims(y_train, axis=1)
columns_val = all_preds_val * np.expand_dims(y_val, axis=1)

all_misses_train = np.where(columns_train <= 0, 1, 0)
all_misses_val = np.where(columns_val <= 0, 1, 0)

data_number_misses_train = np.sum(all_misses_train, axis=0)
data_number_misses_val = np.sum(all_misses_val, axis=0)

# Find datapoints that were miss-classified by n models (BLs)
n_miss_5_train, n_miss_5_val, data_miss_5_train, data_miss_5_val = \
    get_miss_for_n_models(data_number_misses_train, data_number_misses_val, n_models=7)

n_miss_4_train, n_miss_4_val, data_miss_4_train, data_miss_4_val = \
    get_miss_for_n_models(data_number_misses_train, data_number_misses_val, n_models=4)

n_miss_3_train, n_miss_3_val, data_miss_3_train, data_miss_3_val = \
    get_miss_for_n_models(data_number_misses_train, data_number_misses_val, n_models=3)


