import numpy as np
import copy
from typing import Union


class Ensemble:
    """ Container + Builder of final ensemble

        Bagging:
            Prediction of initial ensemble on x-train
            Prediction of initial ensemble on x-validation

            Initial ensemble yh on x-train
            Initial ensemble yh on x-validation

        Parameters:
            y_train (np.ndarray):   Only necessary for multi-class version
            y_val (np.ndarray):     Only necessary for multi-class version

    """

    def __init__(self, x_train, x_val, y_train_comp, y_val_comp, n_start, comb_mech, y_train=None, y_val=None,
                 binary_mode=True):

        self._comb_mech = comb_mech
        self._voting_threshold = 0.01
        self._binary_mode = binary_mode

        # Data
        self.x_train = x_train
        self.x_val = x_val
        self.n_start = n_start

        if binary_mode:
            self.y_train_comp = np.expand_dims(y_train_comp, axis=1)
            self.y_val_comp = np.expand_dims(y_val_comp, axis=1)
        else:
            self.y_train = y_train
            self.y_val = y_val

        # External BLs trained on full train data
        self.external_BLs = []

        # Generation
        self.initial_H = []
        self.initial_preds_on_train = []
        self.initial_preds_on_val = []
        self.initial_columns_on_train = []

        # Bagging initial ensemble
        self.initial_bagging_ens_pred_on_train = None
        self.initial_bagging_ens_column_on_train = None
        self.initial_bagging_ens_column_on_val = None
        self.initial_bagging_ens_pred_on_val = None

        # /Generation

        # MP initial ensemble
        self.initial_mp_ens_pred_on_train = None
        self.initial_mp_ens_column_on_train = None
        self.initial_mp_ens_pred_on_val = None
        self.initial_mp_ens_column_on_val = None
        # /MP initial ensmble

        # Refinement
        self.refinement_H = []
        self.rec_add_refinement_H = []
        self.refinement_preds_on_train = []
        self.refinement_preds_on_val = []
        self.refinement_columns_on_train = []
        # self.final_yh = None
        self._weights = None
        self._model_id = 0
        # /Refinement

        # Final ens prediction
        self.final_ens_pred_on_train = None
        self.final_ens_column_on_train = None
        self.final_ens_column_on_train_binarized = None
        self.final_ens_pred_on_val = None
        self.final_ens_column_on_val = None

        # Combined
        self.binarized_columns = []
        self.new_columns = []

        if comb_mech == 'av':
            self._binarize_preds = False
        else:
            self._binarize_preds = True

    # Define modified protocol for copy.deepcopy(x, memo) to circumvent pickling keras models
    # Only keras models are given by reference
    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            if k != 'initial_H':
                setattr(result, k, copy.deepcopy(v, memodict))
            else:
                setattr(result, k, v)
        return result

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights: np.ndarray):
        self._weights = weights

    @property
    def model_id(self):
        return self._model_id

    @model_id.setter
    def model_id(self, id):
        self._model_id = id

    def add_initial_H(self, initial_model):
        self.initial_H.append(initial_model)

    def add_initial_pred_on_train(self, initial_pred_on_train) -> int:
        """
        - Adds the BL pred to self.initial_pred_on_train
        :param initial_pred_on_train: BL prediction inferred in generation phase
        :return: Error indicator
        """
        self.initial_preds_on_train.append(initial_pred_on_train)

        return 0

    def add_initial_column_on_train(self, initial_yh_on_train):
        self.initial_columns_on_train.append(initial_yh_on_train)
        self.new_columns.append(initial_yh_on_train)

    def add_initial_pred_on_val(self, initial_pred_on_val):
        self.initial_preds_on_val.append(initial_pred_on_val)

    def add_refinement_H(self, refinement_model):
        self.refinement_H.append(refinement_model)
        self.rec_add_refinement_H.append(refinement_model)

    def add_refinement_pred_on_train(self, refinement_pred):
        self.refinement_preds_on_train.append(refinement_pred)

    def add_refinement_pred_on_val(self, refinement_preds):
        self.refinement_preds_on_val.append(refinement_preds)

    def add_refinement_column_on_train(self, refinement_yh_train):
        self.refinement_columns_on_train.append(refinement_yh_train)
        self.new_columns.append(refinement_yh_train)

    def get_initial_preds_train(self) -> np.ndarray:
        """
        - Gets all the BL train predictions in generation phase
        :return: All initial BL train predictions as np.ndarray
        """
        return np.asarray(self.initial_preds_on_train)

    def get_initial_preds_val(self):
        """
        - Gets all the BL validation predictions in generation phase
        :return: All initial BL validation predictions as np.ndarray
        """
        return np.asarray(self.initial_preds_on_val)

    def get_ref_preds_train(self):
        return np.asarray(self.refinement_preds_on_train)

    def get_ref_preds_val(self):
        return np.asarray(self.refinement_preds_on_val)

    def set_initial_bagging_ens_pred_on_train(self, ini_bag_ens_pred_train):
        """
        - Important for RFV-versions, where first iteration is technically computed by bagging
        :param ini_bag_ens_pred_train: : Initial ensemble prediction on train with bagging
        """
        self.initial_bagging_ens_pred_on_train = ini_bag_ens_pred_train

    def set_initial_bagging_ens_pred_on_val(self, ini_bag_ens_pred_val):
        """
        - Important for RFV variants, where first iteration is technically computed by bagging
        :param ini_bag_ens_pred_val: Initial ensemble prediction on validation with bagging
        """
        self.initial_bagging_ens_pred_on_val = ini_bag_ens_pred_val

    def set_initial_bagging_ens_column_on_train(self, ini_bag_ens_col_train):
        """
        - Important for RFV-variants, where first iteration is technically computed by bagging
        :param ini_bag_ens_col_train: Initial ensemble column on train with bagging
        """
        self.initial_bagging_ens_column_on_train = ini_bag_ens_col_train

    def set_initial_bagging_ens_column_on_val(self, ini_bag_ens_col_val):
        """
        - Important for RFV variants, where first iteration is technically computed by bagging
        :param ini_bag_ens_col_val: Initial ensemble column on validarion with bagging
        """
        self.initial_bagging_ens_column_on_val = ini_bag_ens_col_val

    def set_initial_mp_ens_pred_on_train(self, final_ens_pred_on_train):
        self.initial_mp_ens_pred_on_train = final_ens_pred_on_train

    def set_initial_mp_ens_pred_on_val(self, final_ens_pred_on_val):
        self.initial_mp_ens_pred_on_val = final_ens_pred_on_val

    def set_initial_mp_ens_column_on_train(self, final_ens_column_on_train):
        self.initial_mp_ens_column_on_train = final_ens_column_on_train

    def set_initial_mp_ens_column_on_val(self, final_ens_column_on_val):
        self.initial_mp_ens_column_on_val = final_ens_column_on_val

    def get_columns(self, binarize=False):
        """Returns columns in standard format
            binary: (len, 1, 1)
            multi:  """
        if binarize:
            self.binarize_added_yhs()
            return np.asarray(self.binarized_columns)
        else:
            if len(self.refinement_columns_on_train):
                formatted_initial_columns = np.asarray(self.initial_columns_on_train)
                formatted_refinement_columns = np.asarray(self.refinement_columns_on_train)
                return np.concatenate((formatted_initial_columns, formatted_refinement_columns), axis=0)
            else:
                return np.asarray(self.initial_columns_on_train)

    def get_scores(self):
        """ Used in binary and multi mode
            1. Generates ens. final train and val predictions
            2. Generates ens. final train and val columns
        """
        # Calculate column of ensemble in i-th iteration

        if self._binary_mode:
            self.generate_ens_fin_pred_on_train()
            self.generate_ens_fin_pred_on_val()

            # # Mapping to [-1, 1]
            # final_ens_pred_on_train_mapped = -1 + 2 * self.final_ens_pred_on_train
            # final_ens_pred_on_val_mapped = -1 + 2 * self.final_ens_pred_on_val

            column_train = self.final_ens_pred_on_train * self.y_train_comp
            column_val = self.final_ens_pred_on_val * self.y_val_comp
            miss_score_train = sum([1 for yh in column_train if yh <= 0])
            miss_score_val = sum([1 for yh in column_val if yh <= 0])
        else:
            # Ens. output given in shape (J, |M|), [[m,...m_|M|],[]...,[]_|J|]
            self.generate_ens_fin_pred_on_train_multi()
            self.generate_ens_fin_pred_on_val_multi()
            # Train
            # Get shape of fin ens pred and pred for true y
            len_t_t, len_m_t = self.final_ens_pred_on_train.shape
            pred_for_true_y_train = self.final_ens_pred_on_train[range(len_t_t), self.y_train]
            # Generate sol_col
            col_with_hy = np.zeros((len_t_t, len_m_t))
            col_with_hy[range(len_t_t), self.y_train] = pred_for_true_y_train
            # Reduce sol_col to 1D
            col_with_hy_1d = col_with_hy[np.nonzero(col_with_hy)]
            col_with_hy_1d = np.expand_dims(col_with_hy_1d, axis=1)
            # Calculate difference h_y(x) - h_m(x)

            diff_col = -self.final_ens_pred_on_train + col_with_hy_1d
            # Calculate Final
            column_train = col_with_hy + diff_col
            # Calculate miss_score
            truth_matrix = column_train < 0
            miss_score_train = sum([1 if any(row) else 0 for row in truth_matrix])

            # Validation
            # Generate sol_col
            len_t_v, len_m_v = self.final_ens_pred_on_val.shape
            pred_for_true_y_val = self.final_ens_pred_on_val[range(len_t_v), self.y_val]
            col_with_hy_v = np.zeros((len_t_v, len_m_v))
            col_with_hy_v[range(len_t_v), self.y_val] = pred_for_true_y_val
            # Reduce sol_col to 1D
            col_with_hy_1d_v = col_with_hy_v[np.nonzero(col_with_hy_v)]
            col_with_hy_1d_v = np.expand_dims(col_with_hy_1d_v, axis=1)
            # Calculate difference h_y(x) - h_m(x)
            diff_col_v = -self.final_ens_pred_on_val + col_with_hy_1d_v
            # Calculate Final
            column_val = col_with_hy_v + diff_col_v
            # Calculate miss_score
            truth_matrix_val = column_val < 0
            miss_score_val = sum([1 if any(row_val) else 0 for row_val in truth_matrix_val])

        self.final_ens_column_on_train = column_train
        self.final_ens_column_on_val = column_val

        # TODO: Has to be extended to Multi-Class
        metrics_train = None
        metrics_val = None
        # # final_ens_pred: [-1, 1]
        # # y_comp: [-1, 1]
        # # metrics for binary classification
        # if metrics:
        #     y_pred_on_train_comp = \
        #         np.asarray([1 if yh > 0 else -1 for yh in
        #                     self.final_ens_pred_on_train], dtype=np.short)
        #     y_pred_on_val_comp = \
        #         np.asarray([1 if yh > 0 else -1 for yh in
        #                     self.final_ens_pred_on_val], dtype=np.short)
        #     metrics_train = \
        #         {'accuracy': round(
        #             accuracy_score(self.y_train_comp,
        #                            y_pred_on_train_comp), 4),
        #          'precision': round(
        #              precision_score(self.y_train_comp,
        #                              y_pred_on_train_comp), 4),
        #          'recall': round(
        #              recall_score(self.y_train_comp,
        #                           y_pred_on_train_comp), 4),
        #          'f1score': round(
        #              f1_score(self.y_train_comp,
        #                       y_pred_on_train_comp), 4)}
        #     metrics_val = \
        #         {'accuracy': round(
        #             accuracy_score(self.y_val_comp,
        #                            y_pred_on_val_comp), 4),
        #          'precision': round(
        #              precision_score(self.y_val_comp,
        #                              y_pred_on_val_comp), 4),
        #          'recall': round(
        #              recall_score(self.y_val_comp,
        #                           y_pred_on_val_comp), 4),
        #          'f1score': round(
        #              f1_score(self.y_val_comp,
        #                       y_pred_on_val_comp), 4)}

        return miss_score_train, miss_score_val, metrics_train, metrics_val

    ### Helper Methods ###
    def generate_refinement_preds_on_val(self):
        for model in self.rec_add_refinement_H:
            # model.switch_domain()
            val_preds = model.get_prediction(self.x_val, binarize=self._binarize_preds)
            if self._binary_mode:
                val_preds = -1 + 2*val_preds
            self.add_refinement_pred_on_val(val_preds)
        self.rec_add_refinement_H = []

    # Generate binarized yhs on train for each newly added margin
    def binarize_added_yhs(self):
        for new_yh in self.new_columns:
            binarized_yh = np.where(np.asarray(new_yh) > 0, 1, -1)
            self.binarized_columns.append(binarized_yh)
        self.new_columns = []

    def update_voting_threshold(self, n_bls, sig_fac):
        self._voting_threshold = (1/n_bls) * sig_fac

    ### /Helper Methods ###

    def del_last_batch(self):
        self.refinement_H.__delitem__(-1)
        self.refinement_columns_on_train.__delitem__(-1)
        self.refinement_preds_on_train.__delitem__(-1)

    # Initial predictions are aggregated
    def generate_ens_fin_pred_on_train_bagging(self):
        """ Voting + Averaging: Both are weighted by the primal!
            Consider votes of initial preds and new preds
            Multiply the initial weights 1/self._n_start with the weight assigned by the primal LP
            IMPORTANT:      Update the weigths before generating the ensemble preds
        """
        if self.initial_bagging_ens_pred_on_train.ndim != 3:
            self.initial_bagging_ens_pred_on_train = np.expand_dims(self.initial_bagging_ens_pred_on_train, axis=0)
        if self.refinement_preds_on_train:
            all_preds = np.concatenate((self.initial_bagging_ens_pred_on_train, np.asarray(self.refinement_preds_on_train)),
                                       axis=0, dtype=np.double)
        else:
            all_preds = self.initial_bagging_ens_pred_on_train

        if self._comb_mech == 'av':
            self.final_ens_pred_on_train = np.zeros((len(self.x_train), 1))
            idx = 0
            for preds in all_preds:
                self.final_ens_pred_on_train += (preds * self.weights[idx])
                idx += 1
        else:
            zeros_idx = []
            for index, weight in enumerate(self.weights):
                if weight < self._voting_threshold:
                    zeros_idx.append(index)
            all_preds = np.delete(all_preds, zeros_idx, axis=0)
            self.final_ens_pred_on_train = np.array([1 if x > 0 else -1 for x in np.sum(all_preds, axis=0)])
            self.final_ens_pred_on_train = np.expand_dims(self.final_ens_pred_on_train, axis=1)

    # Initial predictions are NOT aggregated
    def generate_ens_fin_pred_on_train(self):
        """ Voting + Averaging: Both are weighted by the MP!
            Consider votes of each individual model
            All weights come from MP
            Generates ens fin pred on train and stores them in attribute
        """

        # initial_preds_on_train_formatted = np.expand_dims(self.initial_preds_on_train, axis=2)
        initial_preds_on_train_formatted = self.initial_preds_on_train
        if self.refinement_preds_on_train:
            all_preds = np.concatenate((initial_preds_on_train_formatted, np.asarray(self.refinement_preds_on_train)),
                                       axis=0, dtype=np.double)
        else:
            all_preds = initial_preds_on_train_formatted

        if self._comb_mech == 'av':
            # Verified!
            weights_formatted = np.expand_dims(self.weights, axis=[1, 2])
            weighted_preds = all_preds * weights_formatted
            self.final_ens_pred_on_train = np.sum(weighted_preds, axis=0)
        else:
            """sng(sum(column)) for ensemble output"""
            zeros_idx = []
            for idx, weight in enumerate(self.weights):
                if weight < self._voting_threshold:
                    zeros_idx.append(idx)
            all_preds = np.delete(all_preds, zeros_idx, axis=0)
            binarized_preds = np.where(all_preds > 0, 1, -1)
            self.final_ens_pred_on_train = np.array([1 if x > 0 else -1 for x in np.sum(binarized_preds, axis=0)])
            self.final_ens_pred_on_train = np.expand_dims(self.final_ens_pred_on_train, axis=1)

        return 0

    # Initial prdictions are aggregated and processed for final preds
    def generate_ens_fin_pred_on_val_bagging(self):
        if self.refinement_H:
            self.generate_refinement_preds_on_val()
            if self._comb_mech == 'av':
                # Verified!
                ini_weighted_preds = np.asarray(self.initial_preds_on_val) * (self.weights[0] / self.n_start)
                self.initial_bagging_ens_pred_on_val = np.sum(ini_weighted_preds, axis=0)

                # ref_preds = []
                # for model in self.refinement_H:
                #     ref_preds.append(model.get_prediction(self.x_val))
                ref_preds = np.asarray(self.refinement_preds_on_val)
                ref_weights_exp = np.expand_dims(self.weights[1:], axis=[1, 2])
                ref_weighted_preds = ref_preds * ref_weights_exp

                all_weighted_preds = np.concatenate((ini_weighted_preds, ref_weighted_preds), axis=0)
                fin_preds_on_val = np.sum(all_weighted_preds, axis=0)
            else:
                # Consider all models as equal and vote the final pred based on their preds
                # all_models = [self.initial_H, *self.refinement_H]
                zeros = []
                cleaned_array = []
                all_preds = [self.initial_preds_on_val, *self.refinement_preds_on_val]
                for index, weight in enumerate(self.weights):
                    if weight < self._voting_threshold:
                        zeros.append(index)
                all_preds = np.delete(all_preds, zeros, axis=0)
                if 0 not in zeros:
                    for ini_h_pred_val in all_preds[0]:
                        cleaned_array.append(ini_h_pred_val)
                    for ref_h_pred in all_preds[1:]:
                        cleaned_array.append(ref_h_pred)
                else:
                    for ref_h_pred in all_preds:
                        cleaned_array.append(ref_h_pred)
                np_cleaned = np.asarray(cleaned_array)
                fin_preds_on_val = np.array([1 if x > 0 else -1 for x in np.sum(np_cleaned, axis=0)])
                fin_preds_on_val = np.expand_dims(fin_preds_on_val, axis=1)
        else:
            # ini_preds = []
            # for ini_model in self.initial_H:
            #     ini_preds.append(ini_model.get_prediction(self.x_val))
            # ini_weighted_preds = np.asarray(ini_preds)                # .squeeze()

            if self._comb_mech == 'av':
                ini_weighted_preds = np.asarray(self.initial_preds_on_val) * (1 / self.n_start)
                self.initial_bagging_ens_pred_on_val = np.sum(ini_weighted_preds, axis=0)
                fin_preds_on_val = self.initial_bagging_ens_pred_on_val
            else:
                zeros_idx = []
                for index, weight in enumerate(self.weights):
                    if weight < self._voting_threshold:
                        zeros_idx.append(index)
                ini_weighted_preds = np.delete(self.initial_preds_on_val, zeros_idx, axis=0)
                ini_weighted_preds = np.sum(ini_weighted_preds, axis=0)
                fin_preds_on_val = np.array([1 if x > 0 else -1 for x in ini_weighted_preds])
                fin_preds_on_val = np.expand_dims(fin_preds_on_val, axis=1)

        self.final_ens_pred_on_val = fin_preds_on_val

        return 0

    # Initial prdictions are NOT aggregated and individually processed for final preds
    def generate_ens_fin_pred_on_val(self):
        """ 1. Generates preds of BLs made in refinement phase on validation data
            2. Generates ens fin pred on val and stores them in attribute"""

        self.generate_refinement_preds_on_val()
        initial_preds_on_val = np.asarray(self.initial_preds_on_val)
        if self.refinement_H:
            ref_preds_on_val = np.asarray(self.refinement_preds_on_val)
            all_preds_on_val = np.concatenate((initial_preds_on_val, ref_preds_on_val), axis=0)
        else:
            all_preds_on_val = initial_preds_on_val

        if self._comb_mech == 'av':
            # Verified
            weights_formatted = np.expand_dims(self.weights, axis=[1, 2])
            all_weighted_preds = all_preds_on_val * weights_formatted
            fin_preds_on_val = np.sum(all_weighted_preds, axis=0)
        else:
            """sng(sum(column)) for ensemble output"""
            # Consider all models as equal and vote the final pred based on their preds
            zeros_idx = []
            for index, weight in enumerate(self.weights):
                if weight < self._voting_threshold:
                    zeros_idx.append(index)
            all_preds_on_val = np.delete(all_preds_on_val, zeros_idx, axis=0)
            binarized_preds = np.where(all_preds_on_val > 0, 1, -1)

            fin_preds_on_val = np.array([1 if x > 0 else -1 for x in np.sum(binarized_preds, axis=0)])
            fin_preds_on_val = np.expand_dims(fin_preds_on_val, axis=1)

        self.final_ens_pred_on_val = fin_preds_on_val

        return 0

    """
    #######################################################
    # Generate Final Ensemble Predictions for Multi-Class #
    #######################################################
    """
    # NOT aggregated
    def generate_ens_fin_pred_on_train_multi(self):

        # initial_preds_on_train_formatted = np.expand_dims(self.initial_preds_on_train, axis=2)
        if self.refinement_preds_on_train:
            all_preds = np.concatenate((self.initial_preds_on_train, np.asarray(self.refinement_preds_on_train)),
                                       axis=0, dtype=np.double)
        else:
            all_preds = self.initial_preds_on_train

        if self._comb_mech == 'av':
            # Verified!
            weights_formatted = np.expand_dims(self.weights, axis=[1, 2])
            weighted_preds = all_preds * weights_formatted
            final_ens_preds_on_train = np.sum(weighted_preds, axis=0)
        else:
            # [[sum(d_0), sum(d_1)...]]
            self.update_voting_threshold(len(all_preds), 0.1)
            zeros_idx = []
            for idx, weight in enumerate(self.weights):
                if weight < self._voting_threshold:
                    zeros_idx.append(idx)
            all_preds = np.delete(all_preds, zeros_idx, axis=0)
            # TODO: Try to implement with np.ndarray instead of Double-Loop
            binarized = []
            for bl_preds in all_preds:
                bl_i = []
                for preds in bl_preds:
                    bin_pred = [1 if x == np.max(preds) else 0 for x in preds]
                    bl_i.append(bin_pred)
                binarized.append(bl_i)
            binarized_preds = np.asarray(binarized)

            final_ens_preds_on_train = np.sum(binarized_preds, axis=0)

        self.final_ens_pred_on_train = final_ens_preds_on_train

        return 0

    def generate_ens_fin_pred_on_val_multi(self):
        self.generate_refinement_preds_on_val()
        initial_preds_on_val = np.asarray(self.initial_preds_on_val)
        if self.refinement_H:
            ref_preds_on_val = np.asarray(self.refinement_preds_on_val)
            all_preds_on_val = np.concatenate((initial_preds_on_val, ref_preds_on_val), axis=0)
        else:
            all_preds_on_val = initial_preds_on_val

        if self._comb_mech == 'av':
            # Verified
            weights_formatted = np.expand_dims(self.weights, axis=[1, 2])
            all_weighted_preds = all_preds_on_val * weights_formatted
            final_ens_preds_on_val = np.sum(all_weighted_preds, axis=0)
        else:
            # [[sum(d_0), sum(d_1)...]]
            self.update_voting_threshold(len(all_preds_on_val), 0.1)
            zeros_idx = []
            for idx, weight in enumerate(self.weights):
                if weight < self._voting_threshold:
                    zeros_idx.append(idx)
            all_preds_on_val_filtered = np.delete(all_preds_on_val, zeros_idx, axis=0)
            # TODO: Try to implement with np.ndarray instead of Double-Loop
            binarized = []
            for bl_preds in all_preds_on_val_filtered:
                bl_i = []
                for preds in bl_preds:
                    bin_pred = [1 if x == np.max(preds) else 0 for x in preds]
                    bl_i.append(bin_pred)
                binarized.append(bl_i)
            binarized_preds = np.asarray(binarized)

            final_ens_preds_on_val = np.sum(binarized_preds, axis=0)

        self.final_ens_pred_on_val = final_ens_preds_on_val

        return 0


class Container(Ensemble):
    def __init__(self, x_train, x_val, n_start, comb_mech, y_train=None, y_val=None,
                 binary_mode=False):
        """
        :param x_train:
        :type x_train:
        :param x_val:
        :type x_val:
        :param n_start: Only relevant for calculating model made by pure bagging
        :type n_start: int
        :param comb_mech:
        :type comb_mech:
        :param y_train:
        :type y_train:
        :param y_val:
        :type y_val:
        :param binary_mode:
        :type binary_mode:
        """
        # super(EnsembleLSB).__init__(x_train, x_val, y_train_comp=None, y_val_comp=None, n_start=n_start,
        #                             comb_mech=comb_mech, y_train=y_train, y_val=y_val, binary_mode=binary_mode)
        super(Container, self).__init__(x_train, x_val, y_train_comp=None, y_val_comp=None, n_start=n_start,
                                        comb_mech=comb_mech, y_train=y_train, y_val=y_val, binary_mode=binary_mode)

        self.transformation_train = []
        self.transformation_val = []
        self.master_model = None
        self.ens_pred_train_hist = []
        self.ens_pred_val_hist = []

        # For debuggin only
        self.ens_model_hist = []

    def get_transformation_train(self, as_array: bool) -> Union[np.ndarray, list]:
        """
        - Getter method returning trans.-train data as either np.ndarray or list
        :param as_array: Determines return type
        :return: Trans.-train as either np.ndarray or list
        """
        if as_array:
            return np.asarray(self.transformation_train)
        return self.transformation_train

    def get_transformation_shape_train(self, as_array) -> Union[tuple, list]:
        """
        - Method to determine the shape of the np.ndarray or the shapes of the individual extractions
        :param as_array: Determines if G can be packed into one array. If not, then feature extraction
                         dimensions of individual BLs differ and as_array is False
        :return: Shape of the np.ndarray as tuple or individual shapes of extractors as list
        """
        if as_array:
            return self.get_transformation_train(as_array=as_array).shape
        else:
            shapes = list()
            for ext in self.get_transformation_train(as_array=as_array):
                shapes.append(ext.shape)
            return shapes

    def get_transformation_val(self, as_array: bool):
        if as_array:
            return np.asarray(self.transformation_val)
        return self.transformation_val

    def get_ens_pred_train_hist(self):
        return np.asarray(self.ens_pred_train_hist)

    def get_ens_pred_val_hist(self):
        return np.asarray(self.ens_pred_val_hist)

    def get_final_ens_pred_on_train(self):
        return self.final_ens_pred_on_train

    def get_final_ens_pred_on_val(self):
        return self.final_ens_pred_on_val

    def add_transformation_train(self, transformation_train_i):
        self.transformation_train.append(transformation_train_i)

    def add_transformation_val(self, transformation_val_i):
        self.transformation_val.append(transformation_val_i)

    def add_to_ens_pred_train_hist(self, ens_pred_train_i):
        self.ens_pred_train_hist.append(ens_pred_train_i)

    def add_to_ens_pred_val_hist(self, ens_pred_val_i):
        self.ens_pred_val_hist.append(ens_pred_val_i)

    def add_to_ens_model_hist(self, ens_model_i):
        self.ens_model_hist.append(ens_model_i)

    def set_final_ens_pred_on_train(self, final_ens_pred_on_train):
        self.final_ens_pred_on_train = final_ens_pred_on_train

    def set_final_ens_pred_on_val(self, final_ens_pred_on_val):
        self.final_ens_pred_on_val = final_ens_pred_on_val

    def set_master_model(self, master_model):
        self.master_model = master_model

    def generate_ens_fin_pred_on_val(self):
        current_fin_model = self.master_model_history[-1]
        self.final_ens_pred_on_val = current_fin_model.get_prediction(self.x_val)

    def get_scores(self):
        """ 1. Getter method to obtain classification output must be obtained
            2. Calculate argmax of output vectors
            3. Compare result in 2. with true labels and return the score
        """
        self.generate_refinement_preds_on_val()
        softmax_labels_train = np.argmax(self.final_ens_pred_on_train)
        softmax_labels_val = np.argmax(self.final_ens_pred_on_val)

        miss_score_train = np.sum(np.where(softmax_labels_train != self.y_train, 1, 0))
        miss_score_val = np.sum(np.where(softmax_labels_val != self.y_val, 1, 0))

        # Reserved for further metrics
        metrics_train = None
        metrics_val = None

        return miss_score_train, miss_score_val, metrics_train, metrics_val



