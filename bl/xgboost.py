import numpy as np
import os
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from bl.bl_base import BLBase


class XGBoost(BLBase):
    def __init__(self, inp,  id, lr=0.1, n_estimators=2, sub_sample=1.0, min_samples_split=2, max_depth=3,
                 tol=0.0001, domain=None, loss=None):
        """ Gradient Boosting base learners. Fitting is time consuming even vor n_estimator=10
        sub_sample:         Subsample interacts with the parameter n_estimators. Choosing subsample < 1.0 leads to a
                            reduction of variance and an increase in bias. Values must be in the range
        min_samples_split:  The minimum number of samples required to split an internal node
        max_depth:          The maximum depth of the individual classification estimators.
        tol:                Minimum change required to run the algorithm
        """
        self._n_estimators = n_estimators
        self._sub_sample = sub_sample
        self._min_samples_split = min_samples_split
        self._max_depth = max_depth
        self._tol = tol
        self._domain = domain
        super(XGBoost, self).__init__(inp, lr, loss, id)

    @classmethod
    def flatten_x(cls, x):
        flattened_x = []
        for i in range(len(x)):
            flattened_x.append(np.concatenate(x[:][i]))
        flattened_x = np.asarray(flattened_x)

        return flattened_x

    def save_model(self, path):
        obj_byte = pickle.dumps(self.model)
        obj_byte = np.asarray(obj_byte)
        np.save(path, obj_byte)

    def build_model(self):
        gradient_boosting_clf = GradientBoostingRegressor(learning_rate=self._lr,
                                                          n_estimators=self._n_estimators,
                                                          subsample=self._sub_sample,
                                                          min_samples_split=self._min_samples_split,
                                                          max_depth=self._max_depth,
                                                          tol=self._tol)

        return gradient_boosting_clf

    def train(self, non_flat_x, y, batch_size=None, epochs=None, callbacks=None):
        flattened_x = non_flat_x
        while flattened_x.ndim > 2:
            flattened_x = self.flatten_x(flattened_x)
        self.model = self.model.fit(flattened_x, y)
        self._retrieve_model_info()
        self.binary_accuracy = self.model.score(flattened_x, y)

    def get_prediction(self, non_flat_x, batch_size=None, binarize=False):
        """Parameters:
                binarize:   Maps the output to {-1,1}; only available in binary mode"""
        flattened_x = non_flat_x
        while flattened_x.ndim > 2:
            flattened_x = self.flatten_x(flattened_x)
        preds = np.expand_dims(self.model.predict(flattened_x), axis=1)
        if self._domain is not None:
            preds = preds * self._domain
        return preds

    def _retrieve_model_info(self):
        s = f'Learning Rate:      {self._lr} \n Loss:     {self._loss} \n ' + \
            f'N Estimators:    {self._n_estimators} \n Sub-Samples:     {self._sub_sample}' + \
            f'Min. Samples Split:    {self._min_samples_split} \n Max. Depth:     {self._sub_sample}'
        self.model_config = s


if __name__ == '__main__':
    curr_dir = os.getcwd()
    main_dir = os.path.dirname(curr_dir)
    dataset = 'cifar10'
    target = 3

    if dataset == 'mnist':
        x_raw = np.load(main_dir + '/dataset/mnist/x.npy')
        y_raw = np.load(main_dir + '/dataset/mnist/y.npy')
        x_raw_val = np.load(main_dir + '/dataset/mnist/x_val.npy')
        y_raw_val = np.load(main_dir + '/dataset/mnist/y_val.npy')
    elif dataset == 'cifar10':
        x_raw = np.load(main_dir + '/dataset/cifar10/x_train_raw.npy')
        # x_raw = np.reshape(x_raw, (x_raw.shape[0], x_raw.shape[1] * x_raw.shape[2] * x_raw.shape[3]))
        y_raw = np.load(main_dir + '/dataset/cifar10/y_train_raw.npy')
        x_raw_val = np.load(main_dir + '/dataset/cifar10/x_val_raw.npy')
        # x_raw_val = np.reshape(x_raw_val, (x_raw_val.shape[0], x_raw_val.shape[1] * x_raw_val.shape[2] * x_raw_val.shape[3]))
        y_raw_val = np.load(main_dir + '/dataset/cifar10/y_val_raw.npy')
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

    x_train = x_raw / 255
    x_val = x_raw_val / 255
    y_train = np.expand_dims(np.where(y_raw == target, 1, -1), axis=1)
    y_val = np.expand_dims(np.where(y_raw_val == target, 1, -1), axis=1)

    inp = (32, 32, 3)
    lr = 0.1
    loss = 'log_loss' # 'deviance', 'exponential'
    id = -1
    xgboost = XGBoost(inp=inp, lr=lr, loss=loss, id=id)
    xgboost.train(non_flat_x=x_train, y=y_train)

    preds_val = xgboost.get_prediction(x_val)
    # preds_val = np.expand_dims(preds_val, axis=1)
    acc = xgboost.binary_accuracy

    yh_val = preds_val * y_val
    miss_val_hist = np.where(yh_val < 0, 1, 0)
    score_val = sum(miss_val_hist)

    print(f'Binary accuracy: {acc}')
    print(score_val)

