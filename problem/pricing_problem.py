import numpy as np
from bl.nn import NN, NNRefine
from bl.ensemble_bls import EnsembleCNNVGG
from bl.cnn import CNN, CNNVGG, CNNLinearLoss, CNNVGGLinearLoss, CNNVGGRelu
from bl.tree import Cart
from bl.xgboost import XGBoost

import logging
logger = logging.getLogger('soda')


class PricingProblem:
    """ Train new model according to reweighted data points from dual.
        Returns model only. Does not generate uyh!
        In the spirirt of Decogo, this class is meant to be a pure container.
    """
    def __init__(self, miss_x: np.ndarray, labels: np.ndarray, bl_type_config, bl_config_ref: tuple, id):
        self.miss_x = miss_x
        self.miss_y = labels
        self.ref_bl_type = bl_type_config
        self.ref_bl_config = bl_config_ref
        self._id = id
        self.inverse_threshold = 0.5

    @staticmethod
    def get_active_data_idx(u: np.ndarray, threshold: float, mode: str):
        if mode == "binary":
            miss_idx = [x for x, y in enumerate(u) if y > threshold]
        elif mode == 'multi':
            u_sum = np.sum(u, axis=1)
            miss_idx, = np.where(u_sum > threshold)
        else:
            raise ValueError(f'No such mode as {mode}')

        return miss_idx

    def update(self, miss_x: np.ndarray, miss_y: np.ndarray, bl_config=None):
        self.miss_x = miss_x
        self.miss_y = miss_y
        self._id += 1
        if bl_config is not None:
            self.ref_bl_config = bl_config

        return 0

    def mlp(self, domain=None, baseline=None):
        """
        :param domain: To be active only for a specified domain
        :type domain: np.ndarray
        :param baseline: For residual learning, is the ensemble prediction
        :type baseline: np.ndarray
        :return: Fitted MLP model
        :rtype: NN
        """
        inp, nn_arch, lr, batch_size, n_episodes, loss, act, metric = self.ref_bl_config
        nn = NN(inp, nn_arch, lr, loss, act, dropout_rates=None, metric=metric, id=self._id, domain=domain,
                baseline_pred=baseline)
        nn.train(self.miss_x, self.miss_y, batch_size, n_episodes, callbacks=None)
        # if nn.binary_accuracy <= self.inverse_threshold:
        #     nn.invert_output(do_invert=True)
        return nn

    def cnn(self, domain=None, baseline=None):
        """
        :param domain: To be active only for a specified domain
        :param baseline: For residual learning, is the ensemble prediction
        :return: Fitted CNN Model with standard or VGG architecture
        :rtype: CNN
        """
        arch, inp, lr, loss, batch_size, n_episodes, metric, nfilter, sfilter, pooling, act, mlp = self.ref_bl_config
        if arch == 'nor':
            cnn = CNN(inp, lr, loss, metric, nfilter, sfilter, pooling, act, mlp, self._id, domain=domain)
        elif arch == 'vgg':
            cnn = CNNVGG(inp, lr, loss, metric, nfilter, sfilter, pooling, act, mlp, self._id, domain=domain)
        elif arch == 'vggrelu':
            cnn = CNNVGGRelu(inp, lr, loss, metric, nfilter, sfilter, pooling, act, mlp, self._id, domain=domain)
        else:
            raise Exception(f'Cnn arch. {arch} can not be identified')
        cnn.train(self.miss_x, self.miss_y, batch_size, n_episodes, callbacks=None)
        # if cnn.binary_accuracy <= self.inverse_threshold:
        #     cnn.invert_output(do_invert=True)

        return cnn

    def fit_enscnnvgg(self):
        inp_dim, lr, loss, batch_size, n_episodes, metric, nfilter, sfilter, pooling, act, mlp = self.ref_bl_config

        # Instantiation involves training, training on train data not
        ens_cnnvgg = EnsembleCNNVGG(inp=inp_dim, lr=lr, loss=loss, metric=metric, nfilter=nfilter, sfilter=sfilter,
                                    pooling_filter=pooling, act=act, mlp=mlp, x_train=self.miss_x, y_train=self.miss_y,
                                    batch_size=batch_size, epochs=n_episodes, domain=None, kernel_initializer=None,
                                    id='Test_EnsBL')

        return ens_cnnvgg

    def tree(self, domain=None, baseline=None):
        """
        :param domain: To be active only for a specified domain
        :type domain: np.ndarray
        :param baseline: For residual learning, is the ensemble prediction
        :type baseline: np.ndarray
        :return: Fitted CART model
        :rtype: Tree
        """
        if len(self.ref_bl_config) == 0:
            tree_clf = Cart(id=self._id, domain=domain)
        else:
            max_depth, max_features, max_leaf_nodes, min_impurity_decrease, \
                class_weight = self.ref_bl_config
            tree_clf = Cart(max_depth, max_features, max_leaf_nodes, min_impurity_decrease,
                            class_weight, self._id, domain=domain)
        tree_clf.train(self.miss_x, self.miss_y)
        # if tree_clf.binary_accuracy <= self.inverse_threshold:
        #     tree_clf.invert_output(do_invert=True)

        return tree_clf

    def fit_xgboost(self, domain=None):
        """
        :param domain: To be active only for a specified domain
        :type domain: np.ndarray
        :param baseline: For residual learning, is the ensemble prediction
        :type baseline: np.ndarray
        :return: Fitted XGBoost model
        :rtype: XGBoost
        """
        inp, lr, n_estimators, sub_sample, min_samples_split, max_depth, tol = self.ref_bl_config
        xgboost_clf = XGBoost(inp=inp, lr=lr, n_estimators=n_estimators, sub_sample=sub_sample,
                              min_samples_split=min_samples_split, max_depth=max_depth, tol=tol,
                              id=str(self._id))

        xgboost_clf.train(self.miss_x, self.miss_y)

        return xgboost_clf

    def solve_pp_data(self):
        switcher = {
            'mlp': self.mlp,
            'cnn': self.cnn,
            'tree': self.tree,
            'xgboost': self.fit_xgboost,
            'enscnnvgg': self.fit_enscnnvgg
        }
        self._id += 1
        func = switcher.get(self.ref_bl_type)
        return func()

    def solve_pp_data_res(self, domain, baseline):
        self._id += 1
        if self.ref_bl_type == 'mlp':
            return self.mlp(domain=domain, baseline=baseline)
        elif self.ref_bl_type == 'cnn':
            return self.cnn(domain=domain, baseline=baseline)
        else:
            return self.tree(domain=domain, baseline=baseline)

    def get_preds_on_r(self, u: np.ndarray, ref_preds: np.ndarray, mode: str):
        pass

    def get_preds_on_non_r(self, u: np.ndarray, ref_preds: np.ndarray, mode: str):
        pass


class PricingProblemLinear(PricingProblem):
    """ Instantiate with all training data"""

    def __init__(self, x: np.ndarray, y: np.ndarray, bl_type_config, bl_config_ref: tuple, id):

        logger.info('Construct LPBoost pricing problem instance')
        self._u = None
        super().__init__(x, y, bl_type_config, bl_config_ref, id)
        self.x_train = x
        self.y_train = y

    def update_u(self, new_u: np.ndarray):
        self._u = new_u

        return 0

    def mlp(self, domain=None, baseline=None):
        inp, nn_arch, lr, batch_size, n_episodes, loss, act, metric = self.ref_bl_config
        nn = NNRefine(inp, nn_arch, lr, loss, act, metric, self._id, domain=domain, baseline_pred=baseline)
        nn.train_with_linear_loss(self.miss_x, self.miss_y, batch_size, n_episodes, u=self._u)
        # if nn.binary_accuracy <= self.inverse_threshold:
        #     nn.invert_output(do_invert=True)
        return nn

    def cnn(self, domain=None, baseline=None):
        arch, inp, lr, loss, batch_size, n_episodes, metric, nfilter, sfilter, pooling, act, mlp = self.ref_bl_config
        if arch == 'nor':
            cnn = CNNLinearLoss(inp, lr, loss, metric, nfilter, sfilter, pooling, act, mlp, self._id, domain=domain)
        elif arch == 'vgg':
            cnn = CNNVGGLinearLoss(inp, lr, loss, metric, nfilter, sfilter, pooling, act, mlp, self._id, domain=domain)
        else:
            raise Exception(f'Cnn arch. {arch} can not be identified')
        cnn.train_with_linear_loss(self.miss_x, self.miss_y, batch_size, n_episodes, u=self._u)
        # if cnn.binary_accuracy <= self.inverse_threshold:
        #     cnn.invert_output(do_invert=True)

        return cnn





