import numpy as np
import os
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import time

curr_dir = os.getcwd()
main_dir = os.path.dirname(curr_dir)
dataset = 'cifar10'
target = 3
binary = False

if dataset == 'mnist':
    x_raw = np.load(curr_dir + '/dataset/mnist/x.npy')
    y_raw = np.load(curr_dir + '/dataset/mnist/y.npy')
    x_raw_val = np.load(curr_dir + '/dataset/mnist/x_val.npy')
    y_raw_val = np.load(curr_dir + '/dataset/mnist/y_val.npy')
elif dataset == 'cifar10':
    x_raw = np.load(curr_dir + '/dataset/cifar10/x_train_raw.npy')
    x_raw = np.reshape(x_raw, (x_raw.shape[0], x_raw.shape[1]*x_raw.shape[2]*x_raw.shape[3]))
    y_raw = np.load(curr_dir + '/dataset/cifar10/y_train_raw.npy')
    x_raw_val = np.load(curr_dir + '/dataset/cifar10/x_val_raw.npy')
    x_raw_val = np.reshape(x_raw_val, (x_raw_val.shape[0], x_raw_val.shape[1] * x_raw_val.shape[2] * x_raw_val.shape[3]))
    y_raw_val = np.load(curr_dir + '/dataset/cifar10/y_val_raw.npy')
    if binary:
        non_target = 2
        target_idx, = np.where(y_raw == target)
        non_target_idx, = np.where(y_raw == non_target)
        all_idx = np.concatenate((target_idx, non_target_idx), axis=0)
        np.random.shuffle(all_idx)
        y_raw = y_raw[all_idx]
        x_raw = x_raw[all_idx]
        target_idx_val, = np.where(y_raw_val == target)
        non_target_idx_val, = np.where(y_raw_val == non_target)
        all_idx_val = np.concatenate((target_idx_val, non_target_idx_val), axis=0)
        np.random.shuffle(all_idx_val)
        y_raw_val = y_raw_val[all_idx_val]
        x_raw_val = x_raw_val[all_idx_val]
else:
    raise NameError(f'Dataset {dataset} not known!')

# Define Train and Validation Data, feature and label according to mode chosen
x_train = x_raw / 255
x_val = x_raw_val / 255
if binary:
    y_train = np.expand_dims(np.where(y_raw == target, 1, -1), axis=1)
    y_val = np.expand_dims(np.where(y_raw_val == target, 1, -1), axis=1)
else:
    y_train = np.expand_dims(y_raw, axis=1)
    y_val = np.expand_dims(y_raw_val, axis=1)

gradient_boosting_clf = GradientBoostingClassifier(n_estimators=2)
start = time.perf_counter()
gradient_boosting_clf.fit(x_train, y_train)
end = time.perf_counter()
duration = end - start

preds_train = gradient_boosting_clf.predict(x_train)
preds_train = np.expand_dims(preds_train, axis=1)
preds_val = gradient_boosting_clf.predict(x_val)
preds_val = np.expand_dims(preds_val, axis=1)

if binary:
    yh_train = preds_train * y_train
    yh_val = preds_val * y_val
    miss_train = sum(np.where(yh_train < 0, 1, 0))
    miss_val = sum(np.where(yh_val < 0, 1, 0))

    print(miss_train)
    print(miss_val)

print(f'Time taken: {duration}')
