"""
Script to test how isolated CNNs perform with given hyper-parameters and dataset
"""
import os
import numpy as np
from bl.ensemble_bls import EnsembleCNNVGG
from utils.data_loader import DataLoader
from keras.metrics import Accuracy
from time import perf_counter


# Data loading parameters
dataset = 'cifar10'
binary_mode = True
target = 3
non_target = 2
curr_dir = os.getcwd()
main_dir = os.path.dirname(curr_dir)

# Load data
n_data_points = 50000
data_loader = DataLoader(dataset_name=dataset, binary_labels=binary_mode, target=target,
                         non_target=non_target, soda_main_dir=main_dir)
x_raw, y_train, x_raw_val, y_val = data_loader.load_data()

# Preprocess data
x_train = x_raw / 255
x_val = x_raw_val / 255
x_train = x_train[:n_data_points]
x_val = x_val[:n_data_points]
y_train = y_train[:n_data_points]
y_val = y_val[:n_data_points]

y_train = np.where(y_train == target, 1, 0)
y_val = np.where(y_val == target, 1, 0)

# EnsembleCNNVGG Hyperparameters
# CNN hyperparameters
inp_dim = (32, 32, 3)
lr = 0.001
loss_fu = 'binary_crossentropy'
batch_s = 50
n_episodes = 1
metric = 1
n_filter = (4, 4)             # (64, 32)
s_filter = (2, 2)
pooling_filter = (2, 2)
act = 'tanh'
mlp = (20, 15, 1)

# Train data
batch_size = 50
epochs = 7


ens_cnnvgg = EnsembleCNNVGG(inp=inp_dim, lr=lr, loss=loss_fu, metric=0, nfilter=n_filter, sfilter=s_filter,
                            pooling_filter=pooling_filter, act=act, mlp=mlp, x_train=x_train, y_train=y_train,
                            batch_size=batch_size, epochs=epochs, domain=None, kernel_initializer=None,
                            id='Test_EnsBL')

pred = ens_cnnvgg.get_prediction(x_train)
print(f'Average variance for all correctly classified datapoitns: {ens_cnnvgg.avg_var_ac}')
print(f'Average variance for all incorrectly classified datapoitns: {ens_cnnvgg.avg_var_ac}')



