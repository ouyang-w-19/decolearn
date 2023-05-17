import numpy as np
import os
import pickle
from sklearn import tree
from bl.bl_base import BLBase


class Cart(BLBase):
    """ Applies binanry trees for base learning

        CART constructs binary trees using the feature and
        threshold that yield the largest information gain at each node.
        Label can NOT be continuous!
        Input: Concateanted X, Label Y
        Output: Predictions
    """

    def __init__(self, max_depth=None, max_features=None, max_leaf_nodes=None,
                 min_impurity_decrease=0.0, class_weight=None, id='0', domain=None, baseline_pred=None):
        self._max_depth = max_depth
        self._max_features = max_features
        self._max_leaf_nodes = max_leaf_nodes
        self._min_impurity_decrease = min_impurity_decrease
        self._class_weight = class_weight
        self._domain = domain
        self._baseline_pred = baseline_pred
        super(Cart, self).__init__(None, None, None, id)
        self.model_config = None

    @classmethod
    def flatten_x(cls, x):
        flattened_x = []
        for i in range(len(x)):
            flattened_x.append(np.concatenate(x[:][i]))
        flattened_x = np.asarray(flattened_x)

        return flattened_x

    @classmethod
    def load_model(cls, path_name: str):
        return pickle.loads(np.load(path_name))

    def save_model(self, path_file):
        obj_byte = pickle.dumps(self.model)
        obj_byte = np.asarray(obj_byte)
        np.save(path_file, obj_byte)

        return 0

    def build_model(self):
        # model = tree.DecisionTreeClassifier(max_depth=self._max_depth, max_features=self._max_features,
        #                                   max_leaf_nodes=self._max_leaf_nodes, min_impurity_decrease=
        #                                   self._min_impurity_decrease, class_weight=self._class_weight)
        model = tree.DecisionTreeRegressor(max_depth=self._max_depth, max_features=self._max_features,
                                           max_leaf_nodes=self._max_leaf_nodes, min_impurity_decrease=
                                           self._min_impurity_decrease)
        return model

    def train(self, non_flat_x, y, batch_size=None, epochs=None, callbacks=None):
        flattened_x = non_flat_x
        while flattened_x.ndim > 2:
            flattened_x = self.flatten_x(flattened_x)
        self.model = self.model.fit(flattened_x, y)
        self._retrieve_model_info()
        self.binary_accuracy = self.model.score(flattened_x, y)

        return 0

    def get_prediction(self, non_flat_x, batch_size=None, binarize=False):
        flattened_x = non_flat_x
        while flattened_x.ndim > 2:
            flattened_x = self.flatten_x(flattened_x)
        preds = np.array([1 if x > 0 else -1 for x in self.model.predict(flattened_x)], dtype=np.double)
        preds = np.expand_dims(preds, axis=1)
        if self._domain is not None:
            preds = preds * self._domain
        # Inverse outputs depending on the deviation from 0.5 in unfavorable direction
        # preds *= preds * self.invert_output
        return preds

    def _retrieve_model_info(self):
        s = f'Tree depth:      {self.model.tree_.max_depth} \n Tree leaves:     {self.model.tree_.n_leaves} \n ' + \
            f'Feature weights:    {self.model.feature_importances_} \n '
        self.model_config = s

        return 0


if __name__ == '__main__':
    import numpy as np
    import os
    import pickle
    import time
    from sklearn import tree
    curr_dir = os.getcwd()
    main_dir = os.path.dirname(curr_dir)
    n_samples = 60000
    binary = False
    dataset = 'cifar10'
    target = 3
    non_target = 2

    if dataset == 'mnist':
        x_raw = np.load(main_dir + '/dataset/mnist/x.npy')
        y_raw = np.load(main_dir + '/dataset/mnist/y.npy')
        x_raw_val = np.load(main_dir + '/dataset/mnist/x_val.npy')
        y_raw_val = np.load(main_dir + '/dataset/mnist/y_val.npy')
    elif dataset == 'cifar10':
        x_raw = np.load(main_dir + '/dataset/cifar10/x_train_raw.npy')
        x_raw = np.reshape(x_raw, (x_raw.shape[0], x_raw.shape[1] * x_raw.shape[2] * x_raw.shape[3]))
        y_raw = np.load(main_dir + '/dataset/cifar10/y_train_raw.npy')
        x_raw_val = np.load(main_dir + '/dataset/cifar10/x_val_raw.npy')
        x_raw_val = np.reshape(x_raw_val,
                               (x_raw_val.shape[0], x_raw_val.shape[1] * x_raw_val.shape[2] * x_raw_val.shape[3]))
        y_raw_val = np.load(main_dir + '/dataset/cifar10/y_val_raw.npy')
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

    clf = Cart()
    start = time.perf_counter()
    clf.train(x_train, y_train)
    end = time.perf_counter()
    preds_train = clf.get_prediction(x_train)
    preds_val = clf.get_prediction(x_val)
    # score_train = clf.score(x_train, y_train)
    # score_val = clf.score(x_val, y_val)

    miss_hist_train = np.where(preds_train != y_train, 1, 0)
    miss_score_train = sum(miss_hist_train)

    miss_hist_val = np.where(preds_val != y_val, 1, 0)
    miss_score_val = sum(miss_hist_val)

    print(f'Time taken: {end-start}')
