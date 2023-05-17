import numpy as np
import os
from bl.umap_bl import UMAPBL
from utils.data_loader import DataLoader
from utils.grapher import Grapher
from keras.metrics import Accuracy

# Instantiating Accuracy
acc = Accuracy()
acc.reset_states()

# Instantiate Grapher
grapher = Grapher()

# Data loading parameters
dataset = 'cifar10'
binary_mode = False
target = None
non_target = None
curr_dir = os.getcwd()
main_dir = os.path.dirname(curr_dir)

# Load data
data_loader = DataLoader(dataset_name=dataset, binary_labels=binary_mode, target=target,
                         non_target=non_target, soda_main_dir=main_dir, targets=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
x_raw, y_train, x_raw_val, y_val = data_loader.load_data()

x_raw = x_raw[:25000]
y_train = y_train[:25000]
x_raw_val = x_raw_val[:25000]
y_val = y_val[:25000]

# Preprocess data
x_train = x_raw / 255
x_val = x_raw_val / 255

# General config
new_dim = 2

# UMAP Config
n_components = new_dim
min_dist = 0.1
local_connectivity = 1.0
metric = 'euclidean'
target_metric = 'categorical'
n_neighbors = 45                    # Old 45
learning_rate = 1.0

# Classifier Configuration
inp = (new_dim,)
layers = (20, 10)
lr_mlp = 0.001
loss = 'sparse_categorical_crossentropy'
act = ('relu', 'softmax')
dropout_rates = (0, 0.3)
metric_mlp = 1
identifier = 'UMAP_Classifier_Test'
batch_size = 50
epochs = 30
classifier_config = (inp, layers, lr_mlp, loss, act, dropout_rates, metric_mlp, epochs, batch_size)


reducer2d = UMAPBL(n_components=n_components,
                   min_dist=min_dist,
                   local_connectivity=local_connectivity,
                   metric=metric,
                   target_metric=target_metric,
                   n_neighbors=n_neighbors,
                   lr_umap=learning_rate,
                   id='Test_UMAP',
                   classifier_config=classifier_config)

# Fit UMAP and UMAP-MLP-Classifier, get trans. and preds. for train
reducer2d.train(x_train, y_train)
transformation_train = reducer2d.get_transformation(x_train)
reducer2d.train_classifier(transformation_train, y_train, batch_size=batch_size, epochs=epochs)
preds_train = reducer2d.get_prediction_classifier(trans=transformation_train)
acc.update_state(y_train, np.argmax(preds_train, axis=1))
accuracy_on_train = acc.result().numpy()
print(f'Accuracy on train is: {accuracy_on_train}')
acc.reset_states()

# Get trans. and preds. for val
transformation_val = reducer2d.get_transformation(x_val)
preds_val = reducer2d.get_prediction_classifier(transformation_val)
acc.update_state(y_val, np.argmax(preds_val, axis=1))
accuracy_on_val = acc.result().numpy()
print(f'Accuracy on val is: {accuracy_on_val}')

# Scatter graph for train data
grapher.make_2d_scatter(axis0=transformation_train[:, 0], axis1=transformation_train[:, 1], labels=y_train,
                        cmap='Spectral', s=5, title='Test')

# Combine train and val data
x_all = np.concatenate((x_train, x_val), axis=0)
y_all = np.concatenate((y_train, y_val), axis=0)

# Scatter graph for train + validation data
transformation_all = reducer2d.get_transformation(x_all)
grapher.make_2d_scatter(axis0=transformation_all[:, 0], axis1=transformation_all[:, 1], labels=y_all,
                        cmap='Spectral', s=5, title='Test')


grapher.show_plots()
