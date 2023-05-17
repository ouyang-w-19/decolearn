import os
import umap
import numpy as np
from utils.xml_processor import XMLProcessor
from utils.grapher import Grapher

curr_dir = os.getcwd()
main_dir = os.path.dirname(curr_dir)
super_dir = os.path.dirname(main_dir)
writer = XMLProcessor(main_dir + '/results')

# Define Path
middle_path = 'MissOnly_vs_RFV/Decomp_Analysis_Fixed_Iter/'

# Load Experiment
xml_name = 'cnn_cnn__cifar10__MissOnly_vs_D3A_3vs2_e7_64-32-32-16_fixed20_nu0.2__r2'
path_to_load = super_dir + '/results/' + middle_path + xml_name + '.xml'
exp_main, exp_ref = writer.load_all(path_to_load, 2)
# exp_ref = writer.load_all(path_to_load)
exp0 = exp_main

# Load Data
complete_path = super_dir + '/results/' + middle_path
x_train = np.load(complete_path + 'x_train.npy') / 255
y_train = np.load(complete_path + 'y_train.npy')
y_train_comp = np.load(complete_path + 'y_train_comp.npy')
x_val = np.load(complete_path + 'x_val.npy') / 255
# y_val = np.load(complete_path + 'y_val.npy')
y_val_comp = np.load(complete_path + 'y_val_comp.npy')


# Preprocess Data
n_features = x_train.shape[1] * x_train.shape[2] * x_train.shape[3]
length_x_train = len(x_train)
x_train = np.reshape(x_train, (length_x_train, n_features))

length_x_val = len(x_val)
x_val = np.reshape(x_val, (length_x_val, n_features))

# Concatenate Train T and Val V + Labels
all_x = np.concatenate((x_train, x_val), axis=0)
y_val = np.where(y_val_comp < 0, 0, 1)
all_y = np.concatenate((y_train, y_val), axis=0)

# Get R for refinement phase
active_sets_data = []
active_sets_label = []
for boosting_iter in exp0['data']['boosting_phase']:
    active_set_u_i = np.asarray(exp0['data']['boosting_phase'][boosting_iter]['u_i'])
    active_set_idx_i, _ = np.where(active_set_u_i > 0)

    active_data_points = x_train[active_set_idx_i]
    active_sets_data.append(active_data_points)

    active_labels_i = y_train[active_set_idx_i]
    active_sets_label.append(active_labels_i)

active_sets_data = np.asarray(active_sets_data)
active_sets_label = np.asarray(active_sets_label)


# Instantiate 2D and 3D reducer
reducer2d = umap.UMAP(n_components=2,
                      min_dist=0.1, # Old: 0.1
                      local_connectivity=1.0,
                      metric='euclidean',
                      target_metric='categorical',
                      n_neighbors=15,   # Old: 15
                      learning_rate=1.0
                      )

reducer3d = umap.UMAP(n_components=3,
                      min_dist=0.1, # Old: 0.1
                      local_connectivity=1.0,
                      metric='euclidean',
                      target_metric='categorical',
                      n_neighbors=15,   # Old: 15
                      learning_rate=1.0
                      )

grapher = Grapher()

# # Fit on Train and create projection for all train data + graph
# title = 'Distribution of training data'
# reducer2d.fit(x_train, y_train)
# projection_train = reducer2d.transform(x_train)
#
# grapher.make_2d_scatter(axis0=projection_train[:, 0], axis1=projection_train[:, 1],
#                         labels=y_train, cmap='Spectral', s=10, title=title)
#
#
# # Show V \cup T on projection that was fitted on train with V having same label as T
# title = 'Distribution of training and validation data - Projection with train only - y_v in Y'
# projection_global = reducer2d.transform(all_x)
# grapher.make_2d_scatter(axis0=projection_global[:, 0], axis1=projection_global[:, 1], labels=all_y,
#                         cmap='Spectral', s=10, title=title)
#
# # Show V \cup T on projection that was fitted on train with V having different labels
# y_val_diff = np.ones((len(y_val_comp),)) * -10
# all_y_v_diff = np.concatenate((y_train, y_val_diff), axis=0)
# title = 'Distribution of training and validation data - Projection with train only - y_v not in Y'
# grapher.make_2d_scatter(axis0=projection_global[:, 0], axis1=projection_global[:, 1], labels=all_y_v_diff,
#                         cmap='Spectral', s=10, title=title)
#
# # Show active set for first and last iteration on projection fitted on train
# title = 'Distribution of active data set - Projection with train - First ref. iteration'
# r_ref_0_x = active_sets_data[0]
# r_ref_0_y = active_sets_label[0]
# projection_r = reducer2d.transform(r_ref_0_x)
# grapher.make_2d_scatter(axis0=projection_r[:,0], axis1=projection_r[:,1], labels=r_ref_0_y,
#                         cmap='Spectral', s=10, title=title)
#
# title = 'Distribution of active data set - Projection with train - Last ref. iteration'
# r_ref_0_x = active_sets_data[-1]
# r_ref_0_y = active_sets_label[-1]
# projection_r = reducer2d.transform(r_ref_0_x)
# grapher.make_2d_scatter(axis0=projection_r[:,0], axis1=projection_r[:,1], labels=r_ref_0_y,
#                         cmap='Spectral', s=10, title=title)
#
# # Show active set for last iteration on projection fitted on R
# title = 'Distribution of active data set - Projection with R - Last ref. iteration'
# r_ref_0_x = active_sets_data[-1]
# r_ref_0_y = active_sets_label[-1]
# projection_r = reducer2d.fit_transform(r_ref_0_x, r_ref_0_y)
# grapher.make_2d_scatter(axis0=projection_r[:,0], axis1=projection_r[:,1], labels=r_ref_0_y,
#                         cmap='Spectral', s=10, title=title)
#
#
# ##############
# ##### 3D #####
# ##############
#
# # Show V \cup T on projection that was fitted on train with V having same label as T in 3D
# title = 'Distribution of training and validation data - Projection with train only - y_v in Y'
# reducer3d.fit(x_train, y_train)
# projection_global_3d = reducer3d.transform(x_train)
# grapher.make_3d_scatter(projection_global_3d[:,0], projection_global_3d[:,1], projection_global_3d[:,2],
#                         labels=y_train, cmap='Spectral', s=2, title=title)
#
# # Show V \cup T on projection that was fitted on train with V having same label as T
# title = 'Distribution of training and validation data - Projection with train only - y_v in Y'
# projection_global_val_3d = reducer3d.transform(all_x)
# grapher.make_3d_scatter(axis0=projection_global_val_3d[:, 0], axis1=projection_global_val_3d[:, 1],
#                         axis2=projection_global_val_3d[:, 2], labels=all_y,
#                         cmap='Spectral', s=2, title=title)
#
# # Show V \cup T on projection that was fitted on train with V having different labels
# y_val_diff = np.ones((len(y_val_comp),)) * -10
# all_y_v_diff = np.concatenate((y_train, y_val_diff), axis=0)
# title = 'Distribution of training and validation data - Projection with train only - y_v not in Y'
# grapher.make_3d_scatter(axis0=projection_global_val_3d[:, 0], axis1=projection_global_val_3d[:, 1],
#                         axis2=projection_global_val_3d[:, 2], labels=all_y_v_diff,
#                         cmap='Spectral', s=2, title=title)
#
# # Show active set for first and last iteration on projection fitted on train
# title = 'Distribution of active data set - Projection with train - First ref. iteration'
# r_ref_0_x = active_sets_data[0]
# r_ref_0_y = active_sets_label[0]
# projection_r_train_3d = reducer3d.transform(r_ref_0_x)
# grapher.make_3d_scatter(axis0=projection_r_train_3d[:,0], axis1=projection_r_train_3d[:,1],
#                         axis2=projection_r_train_3d[:,2], labels=r_ref_0_y,
#                         cmap='Spectral', s=2, title=title)
#
# title = 'Distribution of active data set - Projection with train - Last ref. iteration'
# r_ref_0_x = active_sets_data[-1]
# r_ref_0_y = active_sets_label[-1]
# projection_r_train_3d_last = reducer3d.transform(r_ref_0_x)
# grapher.make_3d_scatter(axis0=projection_r_train_3d_last[:,0], axis1=projection_r_train_3d_last[:,1],
#                         axis2=projection_r_train_3d_last[:,2], labels=r_ref_0_y,
#                         cmap='Spectral', s=2, title=title)
#
# # Show active set for last iteration on projection fitted on R
# title = 'Distribution of active data set - Projection with R - Last ref. iteration'
# r_ref_0_x = active_sets_data[-1]
# r_ref_0_y = active_sets_label[-1]
# projection_r_own_3d_last = reducer3d.fit_transform(r_ref_0_x, r_ref_0_y)
# grapher.make_3d_scatter(axis0=projection_r_own_3d_last[:,0], axis1=projection_r_own_3d_last[:,1],
#                         axis2=projection_r_own_3d_last[:,2], labels=r_ref_0_y,
#                         cmap='Spectral', s=2, title=title)
#
# grapher.show_plots()


########################
##### Multi Labels #####
########################


x_train_all = np.load(main_dir + '/dataset/cifar10/x_train_raw.npy') / 255
y_train_all = np.load(main_dir + '/dataset/cifar10/y_train_raw.npy')
x_val_all = np.load(main_dir + '/dataset/cifar10/x_val_raw.npy') / 255
y_val_all = np.load(main_dir + '/dataset/cifar10/y_val_raw.npy')

# Preprocess Data
n_features_all = x_train_all.shape[1] * x_train_all.shape[2] * x_train_all.shape[3]
length_x_train_all = len(x_train_all)
x_train_all = np.reshape(x_train_all, (length_x_train_all, n_features_all))

length_x_val_all = len(x_val_all)
x_val_all = np.reshape(x_val_all, (length_x_val_all, n_features_all))

t0 = 3
t1 = 8
t2 = 2
t0_train_idx, = np.where(y_train_all == t0)
t1_train_idx, = np.where(y_train_all == t1)
t2_train_idx, = np.where(y_train_all == t2)
all_idx = np.concatenate((t0_train_idx, t1_train_idx, t2_train_idx), axis=0)
np.random.shuffle(all_idx)
x_train_3 = x_train_all[all_idx]
y_train_3 = y_train_all[all_idx]

t0_val_idx, = np.where(y_val_all == t0)
t1_val_idx, = np.where(y_val_all == t1)
t2_val_idx, = np.where(y_val_all == t2)
all_idx_val = np.concatenate((t0_val_idx, t1_val_idx, t2_val_idx), axis=0)
np.random.shuffle(all_idx_val)
x_val_3 = x_val_all[all_idx_val]
y_val_3 = y_val_all[all_idx_val]


all_x_3 = np.concatenate((x_train_3, x_val_3), axis=0)
all_y_3 = np.concatenate((y_train_3, y_val_3), axis=0)

# Calculate projection matrix with train3
title = 'Distribution of training data'
reducer2d.fit(x_train_3, y_train_3)
projection_train_3 = reducer2d.transform(x_train_3)
grapher.make_2d_scatter(axis0=projection_train_3[:, 0], axis1=projection_train_3[:, 1], labels=y_train_3,
                        cmap='Spectral', s=5, title=title)

# Show V \cup T on projection that was fitted on train with V having same label as T
title = 'Distribution of training and validation data - Projection with train only - y_v in Y'
projection_all_x_3 = reducer2d.transform(all_x_3)
grapher.make_2d_scatter(axis0=projection_all_x_3[:, 0], axis1=projection_all_x_3[:, 1], labels=all_y_3,
                        cmap='Spectral', s=5, title=title)

# Show V \cup T on projection that was fitted on train with V NOT having same label as T
title = 'Distribution of training and validation data - Projection with train only - y_v notin Y'
y_val_3_diff = y_val_3 * 0
all_y_3_val_diff = np.concatenate((y_train_3, y_val_3_diff), axis=0)
# projection_all_x_3 = reducer2d.transform(all_x_3)
grapher.make_2d_scatter(axis0=projection_all_x_3[:, 0], axis1=projection_all_x_3[:, 1], labels=all_y_3_val_diff,
                        cmap='Spectral', s=5, title=title)

# Calculate projection matrix with train3
title = 'Distribution of training data'
reducer3d.fit(x_train_3, y_train_3)
projection_train_3 = reducer3d.transform(x_train_3)
grapher.make_3d_scatter(axis0=projection_train_3[:, 0], axis1=projection_train_3[:, 1],
                        axis2=projection_train_3[:, 2], labels=y_train_3, cmap='Spectral', s=10, title=title)

# Show V \cup T on projection that was fitted on train with V having same label as T
title = 'Distribution of training and validation data - Projection with train only - y_v in Y'
projection_all_x_3 = reducer3d.transform(all_x_3)
grapher.make_3d_scatter(axis0=projection_all_x_3[:, 0], axis1=projection_all_x_3[:, 1],
                        axis2=projection_all_x_3[:, 2], labels=all_y_3, cmap='Spectral', s=10, title=title)

# Show V \cup T on projection that was fitted on train with V NOT having same label as T
title = 'Distribution of training and validation data - Projection with train only - y_v notin Y'
y_val_3_diff = y_val_3 * 0
all_y_3_val_diff = np.concatenate((y_train_3, y_val_3_diff), axis=0)
# projection_all_x_3 = reducer2d.transform(all_x_3)
grapher.make_3d_scatter(axis0=projection_all_x_3[:, 0], axis1=projection_all_x_3[:, 1],
                        axis2=projection_all_x_3[:, 2], labels=all_y_3_val_diff, cmap='Spectral', s=10, title=title)
grapher.show_plots()

