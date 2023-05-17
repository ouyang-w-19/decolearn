import os
import keras.datasets as datasets
import numpy as np
from utils.data_loader import DataLoader

soda_dir = os.getcwd()
# Fetch dataset from remote
all_datasets = ('cifar10', 'mnist')
all_loaders = [datasets.cifar10.load_data, datasets.mnist.load_data]
names = ['x_train_raw.npy', 'y_train_raw.npy', 'x_val_raw.npy', 'y_val_raw.npy']

for dataset_i, loader_i in zip(all_datasets, all_loaders):
    (x_train, y_train), (x_val, y_val) = loader_i()
    data = [x_train, y_train.squeeze(), x_val, y_val.squeeze()]

    for data_i, name_i in zip(data, names):
        path_i = soda_dir + '/dataset/' + dataset_i + '/' + name_i
        np.save(path_i, data_i)

# Internal Loader
dataset = 'cifar10'
binary_mode = True
target = 3
non_target = 2
data_loader = DataLoader(dataset_name=dataset, binary_labels=binary_mode, target=target,
                         non_target=non_target, soda_main_dir=soda_dir)

x_raw, y_raw, x_raw_val, y_raw_val = data_loader.load_data(load_old_binary=False)

combined = np.asarray([x_raw, y_raw, x_raw_val, y_raw_val], dtype=object)
data_loader.save_binary_data(combined)

