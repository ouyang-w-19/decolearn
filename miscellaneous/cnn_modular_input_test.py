import os
import numpy as np
from bl.nn import NNModularInput
from utils.data_loader import DataLoader
from bl.cnn import SplittedCNNVGG, SplittedCNNModularInput

# Data loading parameters
dataset = 'cifar10'
binary_mode = False
target = None
non_target = None
curr_dir = os.getcwd()
main_dir = os.path.dirname(curr_dir)

# Load data
n_datapoints = 10000
data_loader = DataLoader(dataset_name=dataset, binary_labels=binary_mode, target=target,
                         non_target=non_target, soda_main_dir=main_dir)
x_raw, y_train, x_raw_val, y_val = data_loader.load_data()

# Preprocess data
x_train = x_raw / 255
x_val = x_raw_val / 255

x_train = x_train[:n_datapoints]
x_val = x_val[:n_datapoints]
y_train = y_train[:n_datapoints]
y_val = y_val[:n_datapoints]

n_bls = 2
cnn_bls = list()
G = list()

# BL config 1
inp1 = (32, 32, 3)
lr1 = 0.001
loss1 = 'sparse_categorical_crossentropy'
metric1 = 1
nfilter1 = (4,)
sfilter1 = (2, 2)
pooling_filter1 = (2, 2)
mlp1 = (20, 10)
extr_act_per_layer1 = ('relu', 'relu')
class_act_per_layer1 = ('relu', 'softmax')

# BL config 2
inp2 = (32, 32, 3)
lr2 = 0.001
loss2 = 'sparse_categorical_crossentropy'
metric2 = 1
nfilter2 = (4, 4)
sfilter2 = (2, 2)
pooling_filter2 = (2, 2)
mlp2 = (20, 10)
extr_act_per_layer2 = ('relu', 'relu')
class_act_per_layer2 = ('relu', 'softmax')

batch_size = 50
epochs = 2

# Master CNN Config.
lr_master = 0.001
loss_master = 'sparse_categorical_crossentropy'
metric_master = 1
# Lower configs
nfilter_lower = 10
sfilter_lower = (3, 3)
pool_type_lower = 'avg'
extractor_act_lower = 'relu'
# Upper configs
nfilter_master = (4, 4)
sfilter_master = ((2, 2), (2, 2))
pooling_filter_master = (('avg', (2, 2)), ('avg', (2, 2)))
extr_act_per_layer_master = ('relu', 'relu')
# Classifier
mlp_master = (20, 15, 10)
class_act_per_layer_master = ('relu', 'relu', 'softmax')


# Train 1st BL
cnn_bl1 = SplittedCNNVGG(inp=inp1, lr=lr1, loss=loss1, metric=metric1, nfilter=nfilter1, sfilter=sfilter1,
                         pooling_filter=pooling_filter1, mlp=mlp1, class_act_per_layer=class_act_per_layer1,
                         extractor_act_per_layer=extr_act_per_layer1, id='BL_1')
cnn_bl1.train(x_train, y_train, batch_size=batch_size, epochs=epochs)
G.append(cnn_bl1.get_transformation(x_train))

# Train 2nd BL
cnn_bl2 = SplittedCNNVGG(inp=inp2, lr=lr2, loss=loss2, metric=metric2, nfilter=nfilter2, sfilter=sfilter2,
                         pooling_filter=pooling_filter2, mlp=mlp2, class_act_per_layer=class_act_per_layer2,
                         extractor_act_per_layer=extr_act_per_layer2, id='BL_2')
cnn_bl2.train(x_train, y_train, batch_size=batch_size, epochs=epochs)
G.append(cnn_bl2.get_transformation(x_train))

inp_master = list()
for g in G:
    inp_master.append(g.shape[1:])

master_cnn = SplittedCNNModularInput(inp=inp_master, lr=lr_master, loss=loss_master, metric=metric_master,
                                     nfilter=nfilter_master, sfilter=sfilter_master,
                                     pooling_filter=pooling_filter_master,
                                     class_act_per_layer=class_act_per_layer_master,
                                     extractor_act_per_layer=extr_act_per_layer_master, mlp=mlp_master,
                                     lower_nfilter=nfilter_lower, lower_sfilter=sfilter_lower, lower_pool_type='avg',
                                     lower_extractor_act=extractor_act_lower, id='test_master')

master_cnn.train(G, y_train, batch_size, epochs)

trans = master_cnn.get_transformation(G, batch_size=None)

print(f'\n Classifier training started: \n')
master_cnn.compile_classifier()
master_cnn.classifier.fit(trans, y_train, batch_size=50, epochs=10)




