import os
import numpy as np
from bl.nn import NNModularInput
from utils.data_loader import DataLoader


# Data loading parameters
dataset = 'cifar10'
binary_mode = False
target = None
non_target = None
curr_dir = os.getcwd()
main_dir = os.path.dirname(curr_dir)

# Load data
data_loader = DataLoader(dataset_name=dataset, binary_labels=binary_mode, target=target,
                         non_target=non_target, soda_main_dir=main_dir)
x_raw, y_train, x_raw_val, y_val = data_loader.load_data()

# Preprocess data
x_train = x_raw / 255
x_val = x_raw_val / 255

# NN Arch
input_dim = (32, 32, 3)
arch = (20, 15, 10)
activations = ['relu'] * 2 + ['softmax']
lr = 0.001
loss = 'sparse_categorical_crossentropy'
metric = 1
identifier = 'NNModularInput_Test'

mlp = NNModularInput(inp=input_dim, layers=arch, act=activations, lr=lr, loss=loss,
                     metric=metric, id=identifier)

mlp.train(x_train, y_train, batch_size=50, epochs=8)
# # Test model if additional input (dummy should have no effect) is implemented in NNModularInput for testing
# mlp.model.fit([x_train, x_train], [y_train], batch_size=None, epochs=8)
# dummy_train_ones = np.ones(shape=(50000, 32, 32, 3))
# dummy_train_halves = dummy_train_ones * 0.5
#
# pred_2xtrain = mlp.model.predict([x_train, x_train])
# pred_xtrain_dummy1 = mlp.model.predict([x_train, dummy_train_ones])
# pred_xtrain_dummy05 = mlp.model.predict([x_train, dummy_train_halves])
#
# tt1 = (pred_2xtrain == pred_xtrain_dummy1)
# tt05 = (pred_2xtrain == pred_xtrain_dummy05)

