"""
Script to test how isolated CNNs perform with given hyper-parameters and dataset
"""
import os
import numpy as np
from bl.cnn import SplittedCNNVGG
from utils.data_loader import DataLoader
from keras.metrics import Accuracy
from time import perf_counter


# Data loading parameters
dataset = 'cifar10'
binary_mode = False
target = None
non_target = None
curr_dir = os.getcwd()
main_dir = os.path.dirname(curr_dir)

# Load data
n_data_points = 10000
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

# CNN hyperparameters
inp_dim = (32, 32, 3)
lr = 0.001
loss_fu = 'sparse_categorical_crossentropy'
batch_s = 50
n_episodes = 1
metric = 1
n_filter = (4,)             # (64, 32)
s_filter = (2, 2)
pooling_filter = (2, 2)
class_act = ['relu', 'softmax']
extra_act = ['relu'] * 2
mlp = (20, 15, 10)

# Instantiate CNN
cnn = SplittedCNNVGG(inp=inp_dim, lr=lr, loss=loss_fu, metric=metric, nfilter=n_filter, sfilter=s_filter,
                     pooling_filter=pooling_filter, class_act_per_layer=class_act,
                     extractor_act_per_layer=extra_act, mlp=mlp, id='TestCNN')

# Train cnn
start = perf_counter()
cnn.train(x_train, y_train, batch_size=batch_s, epochs=n_episodes)
end = perf_counter()

# Get preds
pred_train_vector = cnn.get_prediction(x_train)
pred_train = np.argmax(pred_train_vector, axis=1)

pred_val_vector = cnn.get_prediction(x_val)
pred_val = np.argmax(pred_val_vector, axis=1)

# Instantiate accuracy measurement metric
acc = Accuracy()

# Update acc. object on train data and print train result
acc.update_state(y_train, pred_train)
acc_on_train = acc.result().numpy()
print('\n')
print(f'Train accuracy: {acc_on_train} \n')
acc.reset_states()

# Update acc. object on val data and print val data
acc.update_state(y_val, pred_val)
acc_on_val = acc.result().numpy()
print(f'Val. accuracy: {acc_on_val} \n')

print(f'Time taken: {end-start}')

