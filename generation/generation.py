import keras.metrics
import numpy as np
import matplotlib.pyplot as plt
from bl.nn import NN
from bl.cnn import CNN, CNNVGG, SplittedCNNVGG, CNNVGGRelu
from bl.tree import Cart
from bl.xgboost import XGBoost
from bl.umap_bl import UMAPBL
from bl.ensemble_bls import EnsembleCNNVGG
# from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, f1_score, \
    recall_score

import logging
logger = logging.getLogger('soda')


class Bagging:
    def __init__(self, x_t_pro: np.ndarray, y_t_pro: np.ndarray, y_t_comp_pro, x_v_pro: np.ndarray,
                 y_v, y_v_comp_pro, ini_bl_type: str, ini_bl_config: tuple, n_start: int,
                 comb_mech: str, ens, binary_mode, n_bags=1, bootstrapping=True, debug_mode=False, result_logger=None):
        """
        - Creates the initial ensemble model
        - Averaging: Added model have assigned weight.
        - Voting: Only binary, hard voting; building ens. is always weighted (soft voting)
        - Initial Preds/Ens. Pred: Converted to ndarray in _bagging()
        - Added Preds: As lists
        - Instantiated with correctly formatted data (s. _format_data() in class Ensemble)!
        :param x_t_pro: Features of train data
        :param y_t_pro: Labels of train data
        :param y_t_comp_pro: Only used in binary mode, for generating ensemble with pure bagging
        :param x_v_pro: Features of validation date
        :param y_v: Labels of validation data
        :param y_v_comp_pro: Only used in binary mode, for generating ensemble with pure bagging
        :param ini_bl_type: Initial BL type (cnn, mlp...)
        :param ini_bl_config: Initial BL architecture and hyperparameters
        :param n_start: Number of iterations per bag, ini. BL_{total}=n_start * n_bags
        :param comb_mech: Generation method of ensemble, either voting or averaging
        :param ens: Ensemble to which initial data (BLs, preds,...) is added
        :param binary_mode: If True, then number of unique labels is 2
        :param n_bags: Number of bags in which the dataset is split, BL_{total}=n_start * n_bags
        :param bootstrapping: Determines if random sampling from bags occur
        :param result_logger: Logs data for analysis in self-written class
        """
        self.n_bags = n_bags
        self.bootstrapping = bootstrapping
        self.default_sampling_from_bag = 3
        self.newly_inner_loop = list()

        self.x_train = x_t_pro
        self.y_train = y_t_pro
        self.y_train_comp = y_t_comp_pro
        self.x_val = x_v_pro
        self.y_val = y_v
        self.y_val_comp = y_v_comp_pro

        self._ens = ens

        self.initial_preds_on_val = []
        self.initial_ens_pred_on_val = None
        self.initial_ens_column_on_val = []

        self.initial_bl_type = ini_bl_type.lower()
        self.initial_bl_config = ini_bl_config

        self.n_start = n_start
        self._comb_mech = comb_mech.lower()

        self._model_id = 0
        
        self._rnd_idc = None
        self.initial_H = []
        self._newly_added_bls = None
        self.currently_activated = n_start
        self.initial_preds_on_train = []
        self.initial_ens_pred_on_train = None
        self.initial_columns_on_train = []
        self.initial_ens_column_on_train = None

        self.res_log = result_logger

        self._binary_mode = binary_mode

        if comb_mech not in ['av', 'vo']:
            raise ValueError('Combine mechanism is not defined')

        if comb_mech == 'av':
            self.binarize_preds = False
        else:
            self.binarize_preds = True

        self.acc = keras.metrics.Accuracy()
        self.debug_mode = debug_mode
        self.first_iters = self.add_bls(n_start)

    @property
    def model_id(self):
        return self._model_id

    # Create new sample (\hat{X}, \hat{Y}) from (X,y) with uniform random sampling + replacement
    # def _generate_new_batch(self):
    #     pool_size = len(self.x_train)
    #     self._rnd_idc = np.random.choice(pool_size, size=pool_size, replace=True)
    #     new_x = self.x_train[self._rnd_idc]
    #     new_y = self.y_train[self._rnd_idc]
    #
    #     return new_x, new_y

    def _generate_new_batch(self, bag: tuple) -> tuple:
        """
        - Bootstrap from bag_i
        - If bag = (self.x_train, self.y_train) and bootstrapping = False, this reduces to
          a standard learning problem without bagging
        """
        bag_x, bag_y = bag
        pool_size = len(bag_x)
        self._rnd_idc = np.random.choice(pool_size, size=pool_size, replace=True)
        new_bag_x = bag_x[self._rnd_idc]
        new_bag_y = bag_y[self._rnd_idc]

        return new_bag_x, new_bag_y

    def split_dataset_in_n(self):
        """
        - Splits the dataset in bags of ordered pair U \sub J = X x Y
        - Maintains the label ratios in U
        :return: Returns all the bags as a list of number of specified bag. Each bag contains
                 features and labels
        :rtype: list
        """
        bags = []
        m_classes = np.unique(self.y_train)
        cat_i_split = []
        all_cats = []

        short_length = (self.y_train.shape[0]/m_classes.shape[0]) // self.n_bags
        rest = (self.y_train.shape[0]/m_classes.shape[0])%self.n_bags
        for label in m_classes:
            cat_i_idx, = np.where(self.y_train == label)
            np.random.shuffle(cat_i_idx)
            for n in range(self.n_bags):
                partial_cat = cat_i_idx[0 + n::self.n_bags]
                if rest != 0 and partial_cat.shape[0] == short_length:
                    partial_cat = np.append(partial_cat, partial_cat[-1])
                cat_i_split.append(partial_cat)
            all_cats.append(cat_i_split)
            cat_i_split = []

        all_cats = np.asarray(all_cats)  # (10, 2, 2500)

        bags_idx = np.concatenate(all_cats, axis=1)
        for bag_idx in bags_idx:  # Old: all_cats_t
            np.random.shuffle(bag_idx)
            bag_x = self.x_train[bag_idx]
            bag_y = self.y_train[bag_idx]
            bag = (bag_x, bag_y)
            bags.append(bag)

        return bags

    def add_bls(self, add_kn_bl_per_n_bag: int):
        """
        - Generates initial models and their predictions
        - k is the number to be specified, if n=1, then this reduces to standard modes
            :param add_kn_bl_per_n_bag: k is the number of desired gen. iteration, n is the number of
                                        splits and determines how many BL are generated per gen. iteration
            :return: Error indicator
            :rtype: int
        """
        self.newly_inner_loop = []
        switcher = {
            'mlp': self.fit_mlp,
            'cnn': self.fit_cnn,
            'tree': self.fit_tree,
            'xgboost': self.fit_xgboost,
            'enscnnvgg': self.fit_enscnnvgg
        }
        # If the ensemble has been generated already, reformat the numpy arrays to
        # list to be able to append. This is much more efficient than arr = np.append(arr, app, axis=None)
        if type(self.initial_preds_on_train) is not list:
            self.initial_preds_on_train = list(self.initial_preds_on_train)

        newly_added = []
        logger.info('Construct {0} base learners, '
                    'bl model type: {1}'.format(add_kn_bl_per_n_bag, self.initial_bl_type))

        for i in range(add_kn_bl_per_n_bag):
            bags = self.split_dataset_in_n()
            for bag in bags:
                train_func = switcher.get(self.initial_bl_type)
                model, pred = train_func(bag)

                # Generate initial columns
                if self._binary_mode:
                    pred = -1 + 2 * pred
                    self.initial_preds_on_train.append(pred)
                    column = np.expand_dims(self.y_train_comp, axis=1) * pred
                    # column = np.expand_dims(column, axis=1)
                else:
                    # column = []
                    # Ens. output given in shape (J, |M|), [[m,...m_|M|],[]...,[]_|J|]
                    # Train
                    # Get shape of fin ens pred and pred for true y
                    self.initial_preds_on_train.append(pred)
                    len_t_pred, len_m_pred = pred.shape
                    pred_for_true_y = pred[range(len_t_pred), self.y_train]
                    # Generate sol_col
                    col_with_hy = np.zeros((len_t_pred, len_m_pred))
                    col_with_hy[range(len_t_pred), self.y_train] = pred_for_true_y
                    # Reduce sol_col to 1D
                    col_with_hy_1d = col_with_hy[np.nonzero(col_with_hy)]
                    col_with_hy_1d = np.expand_dims(col_with_hy_1d, axis=1)
                    # Calculate difference h_y(x) - h_m(x)
                    diff_col = -pred + col_with_hy_1d
                    # Calculate Final
                    column = col_with_hy + diff_col

                # column = np.asarray(column)
                self.initial_columns_on_train.append(column)
                # self._ens.add_initial_column_on_train(column)

                # self.initial_H.append(model)
                newly_added.append(model)

                # self._ens.add_initial_pred_on_train(pred)
                # self._ens.add_initial_H(model)
                self.res_log.add_bl_accuracy(model.accuracy)

                # Get accuracy for original train data set
                pred_on_ori_train = -1 + 2 * model.get_prediction(self.x_train)
                column = pred_on_ori_train * np.expand_dims(self.y_train_comp, axis=1)
                miss = np.where(column <= 0, 1, 0)
                bl_acc = 1 - (sum(miss) / len(self.y_train_comp))
                self.res_log.add_bl_accuracy_on_original_train(bl_acc)

                self._model_id += 1
                self.currently_activated = self._model_id

                new_batch = (pred, column, model)
                self.newly_inner_loop.append(new_batch)

        self._newly_added_bls = newly_added
        self.initial_preds_on_train = np.asarray(self.initial_preds_on_train)

        return 0

    @staticmethod
    def get_scores(y_pred_train_comp, y_pred_val_comp, y_train_comp, y_val_comp):

        metrics_train = \
            {'accuracy': round(
                accuracy_score(y_train_comp,
                               y_pred_train_comp), 4),
             'precision': round(
                 precision_score(y_train_comp,
                                 y_pred_train_comp), 4),
             'recall': round(
                 recall_score(y_train_comp,
                              y_pred_train_comp), 4),
             'f1score': round(
                 f1_score(y_train_comp,
                          y_pred_train_comp), 4)}
        metrics_val = \
            {'accuracy': round(
                accuracy_score(y_val_comp,
                               y_pred_val_comp), 4),
             'precision': round(
                 precision_score(y_val_comp,
                                 y_pred_val_comp), 4),
             'recall': round(
                 recall_score(y_val_comp,
                              y_pred_val_comp), 4),
             'f1score': round(
                 f1_score(y_val_comp,
                          y_pred_val_comp), 4)}

        return metrics_train, metrics_val

    def get_bl_preds_on_val(self, one_per_time=None):
        """
        :param one_per_time: If True, then the generation of val pred from newly added BLs is given
                                  one at a time
        :return: newly_added_pred
        """
        if type(self.initial_preds_on_val) is not list:
            self.initial_preds_on_val = list(self.initial_preds_on_val)

        if type(one_per_time) == int:
            curr_model = self._newly_added_bls[one_per_time]
            newly_added_pred = curr_model.get_prediction(self.x_val, binarize=self.binarize_preds)
            if self._binary_mode:
                newly_added_pred = -1 + 2*newly_added_pred
            self.initial_preds_on_val.append(newly_added_pred)
            self._ens.add_initial_pred_on_val(newly_added_pred)

            if self._binary_mode:
                column = newly_added_pred * np.expand_dims(self.y_val_comp, axis=1)
                miss_score = np.where(column <= 0, 1, 0)
                acc_val = 1 - (sum(miss_score) / len(self.y_val_comp))
                self.res_log.add_bl_accuracy_on_original_val(acc_val)
            else:
                self.acc.reset_states()
                self.acc.update_state(self.y_val, np.argmax(newly_added_pred, axis=1))
                self.res_log.add_bl_accuracy_on_original_val(self.acc.result().numpy())
        else:
            for model in self._newly_added_bls:
                newly_added_pred = model.get_prediction(self.x_val, binarize=self.binarize_preds)
                if self._binary_mode:
                    newly_added_pred = -1 + 2 * newly_added_pred
                self.initial_preds_on_val.append(newly_added_pred)
                self._ens.add_initial_pred_on_val(newly_added_pred)
                self.res_log.add_bl_accuracy_on_original_val(newly_added_pred)

        self.initial_preds_on_val = np.asarray(self.initial_preds_on_val)

        return newly_added_pred

    # Determine, how many BLs is used in the initial ensemble
    def max_activated(self, max):
        if 0 <= max <= self._model_id - 1:
            self.currently_activated = max
            return 0
        else:
            print(f'{max} is greater than max available initial models. Too add more initial BLs, use'
                  f'add_bls(n)')
            return 1

    # Helper method. Generates initial ensemble prediction on train set
    def _update_ini_bagging_ens_on_train(self):
        """
        - For binary case only
        - Generate the initial ensemble based on number of activated BLs.
        - With initial ensemble, generate preds on train.
        - Update the ensemble after BLs have been added or removed.
        - Calculates train miss score of initial ensemble based on bagging and stores  it in res_logger
        """
        if self._comb_mech == 'av':
            weight = 1 / self.currently_activated
            # all_preds_weighted = np.expand_dims(self.initial_preds_on_train[:self.currently_activated], axis=2) * weight
            all_preds_weighted = self.initial_preds_on_train[:self.currently_activated] * weight

            self.initial_ens_pred_on_train = np.sum(all_preds_weighted, axis=0)              # Dims: 2

            # self._ens.set_initial_bagging_ens_pred_on_train(self.initial_ens_pred_on_train)

        else:
            binarized_initial_preds = []
            for pred in self.initial_preds_on_train[:self.currently_activated]:
                binarized = np.asarray([1 if x > 0 else -1 for x in pred])
                binarized_initial_preds.append(binarized)
            binarized_initial_preds = np.asarray(binarized_initial_preds)
            votes = np.sum(binarized_initial_preds, axis=0)

            self.initial_ens_pred_on_train = np.asarray([1 if x > 0 else -1 for x in votes])
            self.initial_ens_pred_on_train = np.expand_dims(self.initial_ens_pred_on_train, axis=1)

        self._ens.set_initial_bagging_ens_pred_on_train(self.initial_ens_pred_on_train)

    def _update_ini_bagging_ens_column_on_train(self):
        self.initial_ens_column_on_train = self.initial_ens_pred_on_train * \
                                           np.expand_dims(self.y_train_comp, axis=1)
        self._ens.set_initial_bagging_ens_column_on_train(self.initial_ens_column_on_train)

        miss_train_hist = np.where(self.initial_ens_column_on_train <= 0, 1, 0)
        miss_score_train = sum(miss_train_hist)
        self.res_log.set_initial_bagging_ens_miss_score_train(miss_score_train)

    # Helper method. Generates initial ensemble prediction on validation set
    def _update_ini_bagging_ens_on_val(self):
        """ Description
            - Only for binary task
            - With initial ensemble, generate preds on val
            - Update the ensemble preds on val after BLs have been added or removed
        """
        if self._comb_mech == 'av':
            weight = 1/self.currently_activated
            preds_on_val_n = self.initial_preds_on_val[:self.currently_activated] * weight
            self.initial_ens_pred_on_val = np.sum(preds_on_val_n, axis=0)

            # self._ens.add_initial_ens_pred_on_val(self.initial_ens_pred_on_val)
        else:
            votes = np.sum(self.initial_preds_on_val[:self.currently_activated], axis=0)
            self.initial_ens_pred_on_val = np.asarray([1 if x > 0 else -1 for x in votes])
            self.initial_ens_pred_on_val = np.expand_dims(self.initial_ens_pred_on_val, axis=1)

        self._ens.set_initial_bagging_ens_pred_on_val(self.initial_ens_pred_on_val)

        return 0

    def _update_ini_bagging_ens_column_on_val(self):
        self.initial_ens_column_on_val = self.initial_ens_pred_on_val * np.expand_dims(self.y_val_comp, axis=1)
        self._ens.set_initial_bagging_ens_column_on_val(self.initial_ens_column_on_val)

        miss_val_hist = np.where(self.initial_ens_column_on_val <= 0, 1, 0)
        miss_score_val = sum(miss_val_hist)
        self.res_log.set_initial_bagging_ens_miss_score_val(miss_score_val)

    def update_ini_bagging_ens(self):
        if self._binary_mode:
            self._update_ini_bagging_ens_on_train()
            self._update_ini_bagging_ens_on_val()
            self._update_ini_bagging_ens_column_on_train()
            self._update_ini_bagging_ens_column_on_val()
        else:
            self._update_ini_bagging_ens_on_train_multi()
            self._update_ini_bagging_ens_on_val_multi()
            self._update_ini_bagging_ens_column_on_train_multi()
            self._update_ini_bagging_ens_column_on_val_multi()

    # Generate multilayer perceptron model and fit it on train set
    def fit_mlp(self, bag: tuple):
        if self.bootstrapping:
            new_x, new_y = self._generate_new_batch(bag)
        else:
            new_x, new_y = bag
        inp, nn_arch, lr, batch_size, n_episodes, loss, act, metric = self.initial_bl_config
        nn = NN(inp, nn_arch, lr, loss, act, dropout_rates=None, metric=metric, id=str(self._model_id))
        nn.train(new_x, new_y, batch_size, n_episodes, callbacks=None)
        preds = nn.get_prediction(self.x_train, batch_size=None, binarize=self.binarize_preds)

        return nn, preds

    # Generate convolutional neural network model and fit it on train set
    def fit_cnn(self, bag: tuple):
        if self.bootstrapping:
            new_x, new_y = self._generate_new_batch(bag)
        else:
            new_x, new_y = bag
        arch, inp, lr, loss, batch_size, n_episodes, metric, nfilter, sfilter, pooling, act, mlp = \
            self.initial_bl_config
        if arch == 'nor':
            cnn = CNN(inp, lr, loss, metric, nfilter, sfilter, pooling,  act, mlp, str(self._model_id))
        elif arch == 'vgg':
            cnn = CNNVGG(inp, lr, loss, metric, nfilter, sfilter, pooling, act, mlp, str(self._model_id))
        elif arch == 'vggrelu':
            cnn = CNNVGGRelu(inp, lr, loss, metric, nfilter, sfilter, pooling, act, mlp, str(self._model_id))
        else:
            raise Exception(f'CNN architecture {arch} can not be identified')
        cnn.train(new_x, new_y, batch_size=batch_size, epochs=n_episodes, callbacks=None)
        preds = cnn.get_prediction(self.x_train, batch_size=None, binarize=self.binarize_preds)

        return cnn, preds

    def fit_enscnnvgg(self, bag: tuple):
        if self.bootstrapping:
            new_x, new_y = self._generate_new_batch(bag)
        else:
            new_x, new_y = bag

        inp_dim, lr, loss, batch_size, n_episodes, metric, nfilter, sfilter, pooling, act, mlp = self.initial_bl_config

        # Instantiation involves training, training on train data not
        ens_cnnvgg = EnsembleCNNVGG(inp=inp_dim, lr=lr, loss=loss, metric=metric, nfilter=nfilter, sfilter=sfilter,
                                    pooling_filter=pooling, act=act, mlp=mlp, x_train=new_x, y_train=new_y,
                                    batch_size=batch_size, epochs=n_episodes, domain=None, kernel_initializer=None,
                                    id='Test_EnsBL')
        preds = ens_cnnvgg.get_prediction(self.x_train)

        return ens_cnnvgg, preds

    # Generate tree model and fit it on train set
    def fit_tree(self, bag: tuple):
        if self.bootstrapping:
            new_x, new_y = self._generate_new_batch(bag)
        else:
            new_x, new_y = bag
        if len(self.initial_bl_config) == 0:
            tree_clf = Cart()
        else:
            max_depth, max_features, max_leaf_nodes, min_impurity_decrease, \
            class_weight = self.initial_bl_config
            tree_clf = Cart(max_depth, max_features, max_leaf_nodes, min_impurity_decrease,
                            class_weight, str(self._model_id))

        tree_clf.train(new_x, new_y)
        preds = tree_clf.get_prediction(self.x_train)

        return tree_clf, preds

    def fit_xgboost(self, bag: tuple):
        if self.bootstrapping:
            new_x, new_y = self._generate_new_batch(bag)
        else:
            new_x, new_y = bag
        inp, lr, n_estimators, sub_sample, min_samples_split, max_depth, tol = self.initial_bl_config
        xgboost_clf = XGBoost(inp=inp, lr=lr, n_estimators=n_estimators, sub_sample=sub_sample,
                              min_samples_split=min_samples_split, max_depth=max_depth, tol=tol,
                              id=str(self._model_id))

        xgboost_clf.train(new_x, new_y)
        preds = xgboost_clf.get_prediction(self.x_train)

        return xgboost_clf, preds

    def compare(self):

        if self._binary_mode:
            miss_score_hist_train = np.where(self.initial_ens_column_on_train <= 0, 1, 0)
            miss_score_train = sum(miss_score_hist_train)[0]

            miss_score_hist_val = np.where(self.initial_ens_column_on_val <= 0, 1, 0)
            miss_score_val = sum(miss_score_hist_val)[0]

        else:
            # Calculate miss_score
            miss_score_hist_train = np.where(np.argmax(self.initial_ens_pred_on_train, axis=1) != self.y_train, 1, 0)
            miss_score_train = np.sum(miss_score_hist_train)

            miss_score_hist_val = np.where(np.argmax(self.initial_ens_pred_on_val, axis=1) != self.y_val, 1, 0)
            miss_score_val = np.sum(miss_score_hist_val)

        return miss_score_train, miss_score_val

    def make_graph(self, with_active, ens_miss_scores_train, ens_miss_scores_val):
        plt.figure()
        plt.title('Bagging')
        plt.xlabel('# of BLs')
        plt.ylabel('Miss-classification score')
        plt.plot(with_active, ens_miss_scores_train, linewidth=2, linestyle='dashed', marker='o',
                 color='black', markerfacecolor='black', label='train')
        plt.plot(with_active, ens_miss_scores_val, linewidth=2, linestyle='dashed', marker='o',
                 color='red', markerfacecolor='red', label='validation')
        plt.show()

    # Graphs the miss-classification score as a function of the number of BLs
    def get_bagging_perf(self, with_active=None, graph=False):
        if with_active is None:
            with_active = [i + 1 for i in range(self.currently_activated)]
        cache = self.currently_activated

        for max in with_active:
            if max > self.currently_activated:
                print(f'Error, {max} is greater than max. available models!')
                return 1

        all_yhs = self._ens.get_columns()
        for max in with_active:
            self.currently_activated = max
            self.update_ini_bagging_ens()
            ens_miss_score_train, ens_miss_score_val = self.compare()
            all_yhs_i = all_yhs[:max]
            initial_ens_yh_train_i = self._ens.initial_bagging_ens_column_on_train
            # Create 1D weights array
            weights = [1/max] * max

            # ens_miss_score_train, ens_miss_score_val = ens_miss_score_train[0], ens_miss_score_val[0]
            ens_miss_score_train, ens_miss_score_val = ens_miss_score_train, ens_miss_score_val

            self._ens.weights = weights
            self.res_log.add_alphas_i(weights)

            # Accuracy of ensemble on train
            # self.res_log.add_miss_score_train_i(ens_miss_score_train)
            acc_train = 1 - (ens_miss_score_train / len(self.y_train))
            self.res_log.add_miss_score_train_i(acc_train)

            # self.res_log.add_miss_score_val_i(ens_miss_score_val)
            acc_val = 1 - (ens_miss_score_val / len(self.y_val_comp))
            self.res_log.add_miss_score_val_i(acc_val)

            if self._binary_mode:
                avg_diversity, diversity = \
                    self.res_log.calculate_binary_diversity(all_columns=all_yhs_i,
                                                            final_ensemble_column_train=initial_ens_yh_train_i,
                                                            ens_weights=weights,
                                                            binary=True)
                self.res_log.add_avg_diversity_i(avg_diversity)
                self.res_log.add_diversity_i(diversity)
            else:
                self.res_log.add_avg_diversity_i('Not_Implemented')
                self.res_log.add_diversity_i('Not_Implemented')

        if graph:
            self.make_graph(with_active, self.res_log.miss_score_train_history, self.res_log.miss_score_val_history)

        self.currently_activated = cache

        return 0

    """
    #####################
    #### Multi-class ####
    #####################
    """
    def _update_ini_bagging_ens_on_train_multi(self):

        if self._comb_mech == 'av':
            # Verified!
            weight = 1 / self.currently_activated
            weights_formatted = np.expand_dims(weight, axis=[0, 1, 2])
            weighted_preds = self.initial_preds_on_train[:self.currently_activated] * weights_formatted
            initial_ens_preds_on_train = np.sum(weighted_preds, axis=0)
        else:
            binarized = []
            for bl_preds in self.initial_preds_on_train[:self.currently_activated] :
                bl_i = []
                for preds in bl_preds:
                    max_in_pred = np.max(preds)
                    bin_pred = [1 if x == max_in_pred else 0 for x in preds]
                    bl_i.append(bin_pred)
                binarized.append(bl_i)
            binarized_preds = np.asarray(binarized)

            initial_ens_preds_on_train = np.sum(binarized_preds, axis=0)

        self.initial_ens_pred_on_train = initial_ens_preds_on_train

        return 0

    def _update_ini_bagging_ens_column_on_train_multi(self):
        # Ens. output given in shape (J, |M|), [[m,...m_|M|],[]...,[]_|J|]
        # Train
        # Get shape of fin ens pred and pred for true y
        len_t_t, len_m_t = self.initial_ens_pred_on_train.shape
        pred_for_true_y_train = self.initial_ens_pred_on_train[range(len_t_t), self.y_train]
        # Generate sol_col
        col_with_hy = np.zeros((len_t_t, len_m_t))
        col_with_hy[range(len_t_t), self.y_train] = pred_for_true_y_train
        # Reduce sol_col to 1D
        col_with_hy_1d = col_with_hy[np.nonzero(col_with_hy)]
        col_with_hy_1d = np.expand_dims(col_with_hy_1d, axis=1)
        # Calculate difference h_y(x) - h_m(x)
        diff_col = -self.initial_ens_pred_on_train + col_with_hy_1d
        # Calculate Final
        column_train = col_with_hy + diff_col
        self.initial_ens_column_on_train = column_train

        return 0

    def _update_ini_bagging_ens_on_val_multi(self):
        if self._comb_mech == 'av':
            # Verified!
            weight = 1 / self.currently_activated
            weights_formatted = np.expand_dims(weight, axis=[0, 1, 2])
            weighted_preds = self.initial_preds_on_val * weights_formatted
            initial_ens_preds_on_val = np.sum(weighted_preds, axis=0)
        else:
            binarized = []
            for bl_preds in self.initial_preds_on_val:
                bl_i = []
                for preds in bl_preds:
                    max_in_pred = np.max(preds)
                    bin_pred = [1 if x == max_in_pred else 0 for x in preds]
                    bl_i.append(bin_pred)
                binarized.append(bl_i)
            binarized_preds = np.asarray(binarized)
            initial_ens_preds_on_val = np.sum(binarized_preds, axis=0)

        self.initial_ens_pred_on_val = initial_ens_preds_on_val

        return 0

    def _update_ini_bagging_ens_column_on_val_multi(self):
        # Ens. output given in shape (J, |M|), [[m,...m_|M|],[]...,[]_|J|]
        # Train
        # Get shape of fin ens pred and pred for true y
        len_t_v, len_m_v = self.initial_ens_pred_on_val.shape
        pred_for_true_y_val = self.initial_ens_pred_on_val[range(len_t_v), self.y_val]
        # Generate sol_col
        col_with_hy = np.zeros((len_t_v, len_m_v))
        col_with_hy[range(len_t_v), self.y_val] = pred_for_true_y_val
        # Reduce sol_col to 1D
        col_with_hy_1d = col_with_hy[np.nonzero(col_with_hy)]
        col_with_hy_1d = np.expand_dims(col_with_hy_1d, axis=1)
        # Calculate difference h_y(x) - h_m(x)
        diff_col = -self.initial_ens_pred_on_val + col_with_hy_1d
        # Calculate Final
        column_val = col_with_hy + diff_col

        # Set the initial ens. column
        self.initial_ens_column_on_val = column_val

        return 0


"""
####################
# Enhanced Bagging #
####################
"""


class EnhancedBagging(Bagging):
    """ Not adapted to new """
    def __init__(self, x_t_pro: np.ndarray, y_t_pro: np.ndarray, y_t_comp_pro, x_v_pro: np.ndarray, y_v_pro,
                 y_v_comp_pro, ini_bl_type: str, ini_bl_config: tuple, n_start: int, comb_mech: str,
                 ens, binary_mode, n_bags=1, bootstrapping=False, debug_mode=False, result_logger=None):
        logger.info('Construct enhanced bagging instance')
        self._n_samples = len(x_t_pro)
        self.miss_classified = []
        self.correctly_classified = []
        self.first = True
        super(EnhancedBagging, self).__init__(x_t_pro, y_t_pro, y_t_comp_pro, x_v_pro, y_v_pro, y_v_comp_pro,
                                              ini_bl_type, ini_bl_config, n_start, comb_mech, ens, binary_mode,
                                              n_bags=n_bags, bootstrapping=bootstrapping, debug_mode=debug_mode,
                                              result_logger=result_logger)

    def _generate_new_batch(self, bag):
        if self.first:
            columns = np.ones(self._n_samples) * -1
        else:
            # Modified:
            columns = self.initial_columns_on_train[-1]
            # Unmodified Enhanced Bagging (Able to be parallelized):
            # yhs = self.initial_yhs_on_train[0]

        set_correct_x, set_correct_y, final_incorrect_x, final_incorrect_y = \
            self.calc_correct_and_miss_set(columns)

        set_size_correct = len(set_correct_x)

        if set_size_correct == 0:
            final_x = final_incorrect_x
            final_y = final_incorrect_y
        else:
            final_correct_idx = np.random.choice(set_size_correct, size=set_size_correct, replace=True)
            final_correct_x = set_correct_x[final_correct_idx]
            final_correct_y = set_correct_y[final_correct_idx]

            final_x = np.concatenate((final_incorrect_x, final_correct_x))
            final_y = np.concatenate((final_incorrect_y, final_correct_y))

        self.first = False

        return final_x, final_y

    def calc_correct_and_miss_set(self, columns: np.ndarray):
        if self._binary_mode:
            classified_correctly_idx = [x for x, y in enumerate(columns) if y > 0]
            classified_incorrectly_idx = [x for x, y in enumerate(columns) if y <= 0]

            set_correct_x = self.x_train[classified_correctly_idx]
            set_correct_y = self.y_train[classified_correctly_idx]

            final_incorrect_x = self.x_train[classified_incorrectly_idx]
            final_incorrect_y = self.y_train[classified_incorrectly_idx]
        else:
            truth_matrix_t = columns < 0
            classified_incorrectly_idx = [idx if any(row) else 0 for idx, row in enumerate(truth_matrix_t)]
            classified_correctly_idx = [idx if not any(row) else 0 for idx, row in enumerate(truth_matrix_t)]

            set_correct_x = self.x_train[classified_correctly_idx]
            set_correct_y = self.y_train[classified_correctly_idx]

            final_incorrect_x = self.x_train[classified_incorrectly_idx]
            final_incorrect_y = self.y_train[classified_incorrectly_idx]

        return set_correct_x, set_correct_y, final_incorrect_x, final_incorrect_y

    def make_graph(self, with_active, ens_miss_scores_train, ens_miss_scores_val):
        plt.figure()
        plt.title('Enhanced Bagging')
        plt.xlabel('# of BLs')
        plt.ylabel('Miss-classification score')
        plt.plot(with_active, ens_miss_scores_train, linewidth=2, linestyle='dashed', marker='o',
                 color='black', markerfacecolor='black', label='train')
        plt.plot(with_active, ens_miss_scores_val, linewidth=2, linestyle='dashed', marker='o',
                 color='blue', markerfacecolor='blue', label='validation')
        plt.show()


"""
########################
# Splitted BLs Bagging #
########################
"""


class SplittedBagging(Bagging):
    def __init__(self, x_t_pro: np.ndarray, y_t_pro: np.ndarray, y_t_comp_pro, x_v_pro: np.ndarray,
                 y_v, y_v_comp_pro, ini_bl_type: str, ini_bl_config: tuple, n_start: int,
                 comb_mech: str, ens, binary_mode, n_bags=2, bootstrapping=True,
                 debug_mode=False, result_logger=None):
        """
        - Class used in generation phase to give transformations and predictions separately
        :param x_t_pro: Features of train data
        :param y_t_pro: Labels of traind ata
        :param y_t_comp_pro: Only used in binary mode, for generating ensemble with pure bagging
        :param x_v_pro: Features of validation data
        :param y_v: Labels of validation data
        :param y_v_comp_pro: Only used in binary mode, for generating ensemble with pure bagging
        :param ini_bl_type: Initial BL type (cnn, mlp...)
        :param ini_bl_config: Initial BL architecture and hyperparameters
        :param n_start: Number of iterations per bag, ini. BL_{total}=n_start * n_bags
        :param comb_mech: Generation method of ensemble, either voting or averaging.
        :param ens: Ensemble to which initial data (BLs, preds,...) is added
        :param binary_mode: If True, then number of unique labels is 2
        :param n_bags: Number of bags in which the dataset is split, BL_{total}=n_start * n_bags
        :param bootstrapping: Determines if random sampling from bags occur
        :param result_logger: Logs data for analysis in self-written class
        """
        super(SplittedBagging, self).__init__(x_t_pro, y_t_pro, y_t_comp_pro, x_v_pro,
                                              y_v, y_v_comp_pro, ini_bl_type, ini_bl_config, n_start,
                                              comb_mech, ens, binary_mode, n_bags=n_bags, bootstrapping=bootstrapping,
                                              debug_mode=debug_mode, result_logger=result_logger)

    def add_bls(self, add_kn_bl_per_n_bag):
        """
        - Generates initial models and their predictions
        - k is the number to be specified, if n=1, then this reduces to standard modes
        :param add_kn_bl_per_n_bag: k is the number of desired gen. iteration, n is the number of
                                    splits and determines how many BL are generated per gen. iteration
        :return: Error indicator
        :rtype: int
        """
        self.newly_inner_loop = []
        switcher = {
            'mlp': self.fit_mlp,
            'cnn': self.fit_cnn,
            'umap': self.fit_umap,
            'tree': self.fit_tree,
            'xgboost': self.fit_xgboost
        }
        # If the ensemble has been generated already, reformat the numpy arrays to
        # list to be able to append. This is much more efficient than arr = np.append(arr, app, axis=None)
        if type(self.initial_preds_on_train) is not list:
            self.initial_preds_on_train = list(self.initial_preds_on_train)

        newly_added = []
        logger.info('Construct {0} base learners, '
                    'bl model type: {1}'.format(add_kn_bl_per_n_bag, self.initial_bl_type))

        for i in range(add_kn_bl_per_n_bag):
            bags = self.split_dataset_in_n()
            for iter, bag in enumerate(bags):
                print(f'########################## BL Generation {iter + 1}, Run: {i+1} ##########################')
                train_func = switcher.get(self.initial_bl_type)
                model = train_func(bag)
                trans, pred = self.get_trans_and_pred(model)

                if type(pred).__name__ == 'ndarray':
                    # Generate initial columns
                    if self._binary_mode:
                        pred = -1 + 2 * pred
                        self.initial_preds_on_train.append(pred)
                        column = np.expand_dims(self.y_train_comp, axis=1) * pred
                        # column = np.expand_dims(column, axis=1)
                    else:
                        # column = []
                        # Ens. output given in shape (J, |M|), [[m,...m_|M|],[]...,[]_|J|]

                        # Get shape of fin ens pred and pred for true y
                        # _, new_y = bag
                        len_t_pred, len_m_pred = pred.shape # Divide by number of n_bags
                        # pred_for_true_y = pred[range(len_t_pred), self.y_train]
                        pred_for_true_y = pred[range(len_t_pred), self.y_train]
                        # Generate sol_col
                        col_with_hy = np.zeros((len_t_pred, len_m_pred))
                        col_with_hy[range(len_t_pred), self.y_train] = pred_for_true_y
                        # Reduce sol_col to 1D
                        col_with_hy_1d = col_with_hy[np.nonzero(col_with_hy)]
                        col_with_hy_1d = np.expand_dims(col_with_hy_1d, axis=1)
                        # Calculate difference h_y(x) - h_m(x)
                        diff_col = -pred + col_with_hy_1d
                        # Calculate Final
                        column = col_with_hy + diff_col
                else:
                    column = 'No_classifier_specified'

                # self.initial_H.append(model)
                newly_added.append(model)

                # self._ens.add_initial_pred_on_train(pred)
                # self._ens.add_initial_H(model)
                self.res_log.add_bl_accuracy(model.accuracy)

                self._model_id += 1
                self.currently_activated = self._model_id

                new_batch = (trans, pred, column, model)
                self.newly_inner_loop.append(new_batch)

        self._newly_added_bls = newly_added
        self.initial_preds_on_train = np.asarray(self.initial_preds_on_train)

        return 0

    def get_trans_and_pred(self, wrapper_model) -> tuple:
        """
        - Generates extractions and predictions from the model implemented in the wrapper-class
        - Checks if classifier is configured, if yes, classifier is fitted additionally
        :param wrapper_model: Self-implemented classed in soda.bl with split-architecture
        :return: Transformations and predictions of the wrapper model
        """
        trans = wrapper_model.get_transformation(self.x_train)

        # For models having a classifier (e.g. UMAP-BLs classifier is optional)
        if self.initial_bl_type == 'umap':
            _, _, _, _, _, _, _, _, classifier_config = self.initial_bl_config
            _, classifier_config_tuple = zip(*classifier_config.items())
            _, _, _, _, _, _, _, epochs, batch_size = classifier_config_tuple
            wrapper_model.train_classifier(trans, self.y_train, epochs=epochs, batch_size=batch_size)
        pred = wrapper_model.get_prediction_classifier(trans)

        return trans, pred

    def fit_mlp(self, bag: tuple):
        pass

    # Generate convolutional neural network model with splitted data structure
    def fit_cnn(self, bag: tuple):
        if self.bootstrapping:
            new_x, new_y = self._generate_new_batch(bag)
        else:
            new_x, new_y = bag
        _, arch, inp, lr, loss, batch_size, n_episodes, metric, nfilter, sfilter, pooling, class_act, \
            extra_act, mlp = self.initial_bl_config
        cnnvgg_s = SplittedCNNVGG(inp, lr, loss, metric, nfilter, sfilter, pooling, class_act, extra_act, mlp,
                                  str(self._model_id))
        cnnvgg_s.train(new_x, new_y, batch_size=batch_size, epochs=n_episodes, callbacks=None)

        # transformation = cnnvgg_s.get_transformation(new_x)
        # prediction = cnnvgg_s.get_prediction_classifier(transformation)

        return cnnvgg_s

    def fit_umap(self, bag: tuple):
        """
        - Fit the supervised UMAP-BL Transformer on the bag specified.
        :param bag: Disjoint subset containing features and labels; balances
        :return: Fitted UMAP-Wrapper with possibly fitted classifier
        """
        print('Fitting UMAP-BL...')
        if self.bootstrapping:
            new_x, new_y = self._generate_new_batch(bag)
        else:
            new_x, new_y = bag
        _, n_components, min_dist, local_connectivity, metric, target_metric, \
            n_neighbors, learning_rate, classifier_config = self.initial_bl_config
        if classifier_config:
            _, classifier_config_tuple = zip(*classifier_config.items())
            classifier_config = classifier_config_tuple
        umap_bl = UMAPBL(n_components, min_dist, local_connectivity, metric, target_metric,
                         n_neighbors, learning_rate, str(self._model_id), classifier_config=classifier_config)
        umap_bl.train(new_x, new_y)

        return umap_bl

    def fit_tree(self, bag: tuple):
        pass

    def fit_xgboost(self, bag: tuple):
        pass

    def get_bl_preds_on_val(self, one_per_time=None):
        """
        :param      one_per_time: If True, then the generation of val pred from newly added BLs is given
                                  one at a time
        :type       one_per_time:  Union[None, int]
        :return:    newly_added_pred in one_per_time mode, list of newly added preds in loop mode
        :rtype:     Union[np.ndarray, list]
        """
        if type(self.initial_preds_on_val) is not list:
            self.initial_preds_on_val = list(self.initial_preds_on_val)

        if type(one_per_time) == int:
            curr_model = self._newly_added_bls[one_per_time]
            newly_added_trans = curr_model.get_transformation(self.x_val, binarize=self.binarize_preds)
            newly_added_pred = curr_model.get_prediction_classifier(newly_added_trans)
            if self._binary_mode and newly_added_pred:
                newly_added_pred = -1 + 2*newly_added_pred
        else:
            newly_added_trans_list = []
            newly_added_pred_list = []
            for model in self._newly_added_bls:
                model: SplittedCNNVGG
                newly_added_trans = model.get_transformation(self.x_val, batch_size=None)
                newly_added_trans_list.append(newly_added_trans)
                newly_added_pred = model.get_prediction_classifier(newly_added_trans, batch_size=None,
                                                                   binarize=self.binarize_preds)
                if self._binary_mode:
                    newly_added_pred = -1 + 2 * newly_added_pred
                newly_added_pred_list.append(newly_added_pred)

            newly_added_trans = newly_added_trans_list
            newly_added_pred = newly_added_pred_list

        return newly_added_trans, newly_added_pred


"""
###########################
# Test Generation Classes #
###########################
"""
if __name__ == '__main__':
    import os
    from soda.ensemble.ensemble import Container
    from soda.ls_boosting.lsb_run import cnn_cnn_mlp as exp_config, DataLoader, ResultLogger

    main_dir = os.path.dirname(os.getcwd())

    loader = DataLoader(dataset_name='cifar10', soda_main_dir=main_dir, binary_labels=False)
    x_raw, y_train, x_raw_val, y_val = loader.load_data()
    x_train = x_raw / 255
    x_val = x_raw_val / 255

    res_logger = ResultLogger()

    ens = Container(x_train, x_val, exp_config['n_start'], exp_config['comb_me'],
                    y_train=y_train, y_val=y_val, binary_mode=False)

    _, initial_bl_config = zip(*exp_config['bl_config'].items())

    bagging = SplittedBagging(x_train, y_train, None, x_val,
                              y_val, None, exp_config['bl_type'], initial_bl_config,
                              exp_config['n_start'], exp_config['comb_me'], ens,
                              exp_config['binary_mode'], n_bags=exp_config['n_bags'],
                              bootstrapping=False, result_logger=res_logger)

