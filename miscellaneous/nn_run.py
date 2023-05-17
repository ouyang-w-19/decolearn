import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from utils.data_loader import DataLoader
from keras.metrics import Accuracy
from bl.nn import NN, NNFunc, NNModularInput
from time import perf_counter

# Instantiating Accuracy
acc = Accuracy()
acc.reset_states()

# Data loading parameters
dataset = 'mnist'
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

# Model conf.
inp_dim = (28, 28)
layers = (35, 25, 15, 10)
lr = 0.001
loss = 'sparse_categorical_crossentropy'
act = ['relu'] * 3 + ['softmax']
dropouts = [0.1, 0.1, 0.1, 0.0]
metric = 1
identifier = 'test_model'

# Train config.
epochs = 10
batch = 50

nn_drop = NN(inp=inp_dim, layers=layers, lr=lr, loss=loss, act=act, dropout_rates=dropouts,
             metric=metric, id=identifier)
# nn = NNFunc(inp=inp_dim, layers=layers, lr=lr, loss=loss, act=act, dropout_rates=None, metric=metric, id=identifier)
nn = NNModularInput(inp=inp_dim, layers=layers, lr=lr, loss=loss, act=act, dropout_rates=dropouts,
                    metric=metric, id=identifier)

# Train the network
start_nn_drop_train = perf_counter()
nn_drop.train(x_train, y_train, batch_size=batch, epochs=epochs)
end_nn_drop_train = perf_counter()
pred_nn_drop_val = nn_drop.get_prediction(x_val)

start_nn_train = perf_counter()
nn.train(x_train, y_train, batch_size=batch, epochs=epochs)
end_nn_train = perf_counter()
pred_nn_val = nn.get_prediction(x_val)

# Print Results
acc.update_state(y_val, np.argmax(pred_nn_drop_val, axis=1))
print(f'\n \n ')
print(f'----NN_Drop prediction for val: {acc.result().numpy()}')
acc.reset_states()
acc.update_state(y_val, np.argmax(pred_nn_val, axis=1))
print(f'----NN prediction for val: {acc.result().numpy()}')
acc.reset_states()

print(f'----Time taken for nn_drop: {end_nn_drop_train - start_nn_drop_train} \n')
print(f'----Time taken for nn_    : {end_nn_train - start_nn_train} \n')
# input_dims = (28, 28)
# learning_rate = 0.001
# metric = 0
# loss = 'binary_crossentropy'
# mlp = (4, 1)
# epochs = 15
# output_activation_name = ['tanh', 'sigmoid']
# label_type_name = ['0 or 1', '-1 or 1']
#

# for metric in [0, 1]:
#     # metric = 0, output layer uses tanh activation function
#     # metric = 1, output layer uses sigmoid activation function
#     for label_type in [1]:
#         # label type 0: training label = {0, 1}
#         # label type 1: training label = {-1, 1}
#         nn = NN(input_dims, mlp, learning_rate, loss, metric, str(1))
#         if label_type == 0:
#             nn.train(x_train, y_train, 50, epochs)
#         else:
#             nn.train(x_train, y_train, 50, epochs)
#         y_pred_rate = nn.get_prediction(x_train)
#         plt.hist(y_pred_rate, bins='auto')
#
#         plt.title("Hist of MLP output: output layer "
#                   "activation - {0}; label type - {1}".format(
#                    output_activation_name[metric],
#                    label_type_name[label_type]))
#
#         plt.show()
#         print("output layer "
#               "activation - {0}; label type - {1}".format(
#                    output_activation_name[metric],
#                    label_type_name[label_type]))
#         for threshold in [0, 0.5]:
#             m = tf.keras.metrics.BinaryAccuracy(threshold=threshold)
#             m.update_state(y_train, y_pred_rate)
#             print('---------Accuracy with '
#                   'threshold {0}: {1}'.format(threshold,
#                                               m.result().numpy()))
