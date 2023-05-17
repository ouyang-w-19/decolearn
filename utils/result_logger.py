import numpy as np
from typing import Tuple


class ResultLogger:
    def __init__(self):
        """
        - Serves as a namespace for results and performs analysis tasks on the result data
        """
        # Logger Info #
        self.u_history = []
        self.n_active_u = []
        self.mp_history = []
        self.margin = []
        self.alpha_history = []
        self.miss_score_train_history = []
        self.miss_score_train_final_bagging = None
        self.miss_score_val_history = []
        self.miss_score_val_final_bagging = None
        self.avg_diversity_history = []
        self.diversity_history = []
        self.average_bl_accuracy = -1
        self.bl_accuracies = []
        self.bl_accuracies_on_original_train = []
        self.bl_accuracies_on_original_val = []
        self.perf_time = -1
        self.initial_bagging_ens_miss_score_train = -1
        self.initial_bagging_ens_miss_score_val = -1

        # accuracy, precision, recall, f1score
        self.metric_train_history = {}
        self.metric_train_final_bagging = {}
        self.metric_val_history = {}
        self.metric_val_final_bagging = {}

        # Accuracy of T\R
        self.ref_bl_column_on_non_r = []
        self.ref_bl_column_on_r = []

        # Pred on Train and Val
        self.initial_preds_on_train = None
        self.initial_preds_on_val = None
        self.ref_preds_on_train = None
        self.ref_preds_on_val = None

        # Alternative mp ens @ iter = n_st
        self.miss_mp_initial_ens_train = None
        self.miss_mp_initial_ens_val = None
        self._gen_end = None

        # Indicate if there is only one y-label
        self.multiple_y_history = []

        # Stores the extractor output shape
        self.extractor_output_shape_hist = []

        # Stores the accuracies of BLs in gen. and ref. phase
        self.initial_accuracy_on_train_history = []
        self.initial_accuracy_on_val_history = []
        self.ref_accuracy_on_train_history = []
        self.ref_accuracy_on_val_history = []
        self.ens_accuracy_on_train_history = []
        self.ens_accuracy_on_val_history = []

        # Performance time split into gen. and ref. phases
        self.gen_time = 0
        self.ref_time = 0

        # Mapping of all results, needs to be updated with each extension
        self.all_results = None

    @property
    def gen_end(self):
        return self._gen_end

    @gen_end.setter
    def gen_end(self, phase_end: int):
        self._gen_end = phase_end

    def add_u_i(self, u) -> int:
        self.u_history.append(u)
        return 0

    def add_to_n_active_u_history(self, active_u) -> int:
        self.n_active_u.append(active_u)
        return 0

    def add_mp_val_i(self, mp_val) -> int:
        self.mp_history.append(mp_val)
        return 0

    def add_alphas_i(self, alpha) -> int:
        self.alpha_history.append(alpha)
        return 0

    def add_miss_score_train_i(self, miss_score_train) -> int:
        self.miss_score_train_history.append(miss_score_train)
        return 0

    def add_miss_score_val_i(self, miss_score_val) -> int:
        self.miss_score_val_history.append(miss_score_val)
        return 0

    def add_metrics_val_i(self, metrics) -> int:
        if metrics:
            for keys, values in metrics.items():
                if keys in self.metric_val_history.keys():
                    self.metric_val_history[keys].append(values)
                else:
                    self.metric_val_history[keys] = [values]
        return 0

    def add_metrics_train_i(self, metrics) -> int:
        if metrics:
            for keys, values in metrics.items():
                if keys in self.metric_train_history.keys():
                    self.metric_train_history[keys].append(values)
                else:
                    self.metric_train_history[keys] = [values]
        return 0

    def add_avg_diversity_i(self, avg_diversity) -> int:
        self.avg_diversity_history.append(avg_diversity)
        return 0

    def add_diversity_i(self, diversity) -> int:
        self.diversity_history.append(diversity)
        return 0

    def add_bl_accuracy(self, bl_accuracy) -> int:
        self.bl_accuracies.append(bl_accuracy)
        return 0

    def add_ref_bl_column_on_non_r(self, non_r_pred) -> int:
        self.ref_bl_column_on_non_r.append(non_r_pred)
        return 0

    def add_ref_bl_column_on_r(self, r_pred) -> int:
        self.ref_bl_column_on_r.append(r_pred)
        return 0

    def add_margin(self, margin) -> int:
        self.margin.append(margin)
        return 0

    def add_multiple_y_indicator(self, multiple_y_indicator: bool) -> int:
        self.multiple_y_history.append(multiple_y_indicator)
        return 0

    def add_to_extractor_output_shape_history(self, extractor_output_shape) -> int:
        self.extractor_output_shape_hist.append(extractor_output_shape)
        return 0

    def add_to_initial_accuracy_on_train_history(self, initial_accuracy_on_train) -> int:
        self.initial_accuracy_on_train_history.append(initial_accuracy_on_train)
        return 0

    def add_to_initial_accuracy_on_val_history(self, initial_accuracy_on_val) -> int:
        self.initial_accuracy_on_val_history.append(initial_accuracy_on_val)
        return 0

    def add_to_ref_accuracy_on_train_history(self, ref_accuracy_on_train) -> int:
        self.ref_accuracy_on_train_history.append(ref_accuracy_on_train)
        return 0

    def add_to_ref_accuracy_on_val_history(self, ref_accuracy_on_val) -> int:
        self.ref_accuracy_on_val_history.append(ref_accuracy_on_val)
        return 0

    def add_to_ens_accuracy_on_train_history(self, ens_accuracy_on_train) -> int:
        self.ens_accuracy_on_train_history.append(ens_accuracy_on_train)
        return 0

    def add_to_ens_accuracy_on_val_history(self, ens_accuracy_on_val) -> int:
        self.ens_accuracy_on_val_history.append(ens_accuracy_on_val)
        return 0

    def add_bl_accuracy_on_original_train(self, bl_acc_train: np.ndarray):
        """
        :param bl_acc_train: BL accuracy for the original training datasete
        """
        self.bl_accuracies_on_original_train.append(bl_acc_train)

    def add_bl_accuracy_on_original_val(self, bl_acc_val: np.ndarray):
        """
        :param bl_acc_val: BL accuracy for the original validation dataset
        """
        self.bl_accuracies_on_original_val.append(bl_acc_val)

    def set_algo_perf_time(self, perf_time: float) -> int:
        self.perf_time = perf_time
        return 0

    def set_gen_perf_time(self, gen_time: float) -> int:
        self.gen_time = gen_time
        return 0

    def set_ref_perf_time(self, ref_time: float) -> int:
        self.ref_time = ref_time
        return 0

    def set_initial_bagging_ens_miss_score_train(self, initial_bagging_ens_miss_score_train) -> int:
        self.initial_bagging_ens_miss_score_train = initial_bagging_ens_miss_score_train
        return 0

    def set_initial_bagging_ens_miss_score_val(self, initial_bagging_ens_miss_score_val) -> int:
        self.initial_bagging_ens_miss_score_val = initial_bagging_ens_miss_score_val
        return 0

    def set_initial_preds_on_train(self, ini_preds_train) -> int:
        self.initial_preds_on_train = ini_preds_train
        return 0

    def set_initial_preds_on_val(self, ini_preds_val) -> int:
        self.initial_preds_on_val = ini_preds_val
        return 0

    def set_refinement_preds_on_train(self, ref_preds_on_train) -> int:
        self.ref_preds_on_train = ref_preds_on_train
        return 0

    def set_refinement_preds_on_val(self, ref_preds_on_val) -> int:
        self.ref_preds_on_val = ref_preds_on_val
        return 0

    # Methods for alternative mp ens @ iter = n_st
    def calculate_miss_mp_initial_ens_train(self, alt_ens_train_score) -> int:
        alt_ens_train_score_array = np.where(alt_ens_train_score <= 0, 1, 0)
        self.miss_mp_initial_ens_train, = sum(alt_ens_train_score_array)
        return 0

    def calculate_miss_mp_initial_ens_val(self, alt_ens_val) -> int:
        alt_ens_val_score_array = np.where(alt_ens_val <= 0, 1, 0)
        self.miss_mp_initial_ens_val, = sum(alt_ens_val_score_array)
        return 0

    def get_results(self) -> tuple:
        """
        :return: A tuple containing LP-Boost result values
        """
        self.calculate_average_bl_accuracy()
        results = self.mp_history, self.miss_score_train_history, self.miss_score_val_history, \
            self.avg_diversity_history, self.diversity_history, self.alpha_history, \
            self.miss_mp_initial_ens_train, self.miss_mp_initial_ens_val, self._gen_end, self.bl_accuracies, \
            self.average_bl_accuracy, self.perf_time, self.ref_bl_column_on_non_r, self.ref_bl_column_on_r, \
            self.u_history, self.initial_preds_on_train, self.initial_preds_on_val, self.ref_preds_on_train, \
            self.ref_preds_on_val, self.margin, self.multiple_y_history, self.bl_accuracies_on_original_train, \
            self.bl_accuracies_on_original_val
        return results

    def get_selected_results(self, selection: Tuple[str]) -> tuple:
        """
        :param selection: Selected values to be returned for storage
        :return: Tuple containing the selected result values
        """
        return tuple([self.all_results[key] for key in selection])

    def update_attribute_map(self) -> int:
        """
        - Updates attribute map to account for changes in its values
        :return: Error indicator
        """
        self.all_results = {
                            'mp': self.mp_history,
                            'miss_score_train': self.miss_score_train_history,
                            'miss_score_val': self.miss_score_val_history,
                            'avg_diversity': self.avg_diversity_history,
                            'diversity': self.diversity_history,
                            'alpha': self.alpha_history,
                            'miss_mp_initial_ens_train': self.miss_mp_initial_ens_train,
                            'miss_mp_initial_ens_val': self.miss_mp_initial_ens_val,
                            'gen_end': self.gen_end,
                            'bl_accuracies': self.bl_accuracies,
                            'avg_bl_accuracy': self.average_bl_accuracy,
                            'gen_time': self.gen_time,
                            'ref_time': self.ref_time,
                            'perf_time': self.perf_time,
                            'ref_bl_column_on_non_r': self.ref_bl_column_on_non_r,
                            'ref_bl_column_on_r': self.ref_bl_column_on_r,
                            'n_miss': self.n_active_u,
                            'u': self.u_history,
                            'initial_preds_on_train': self.initial_preds_on_train,
                            'initial_preds_on_val': self.initial_preds_on_val,
                            'ref_preds_on_train': self.ref_preds_on_train,
                            'ref_preds_on_val': self.ref_preds_on_val,
                            'maring': self.margin,
                            'multiple_y': self.multiple_y_history,
                            'extractor_output_shape': self.extractor_output_shape_hist,
                            'initial_accuracy_on_train': self.initial_accuracy_on_train_history,
                            'initial_accuracy_on_val': self.initial_accuracy_on_val_history,
                            'ref_accuracy_on_train': self.ref_accuracy_on_train_history,
                            'ref_accuracy_on_val': self.ref_accuracy_on_val_history,
                            'ens_accuracy_on_train': self.ens_accuracy_on_train_history,
                            'ens_accuracy_on_val': self.ens_accuracy_on_val_history
                          }
        return 0

    ##############################
    ####### Analysis Tasks #######
    ##############################

    # Diversity Helper Functions
    def binarize_ens_column_on_train(self, final_ens_yh_train):
        final_ens_yh_on_train_binarized = np.where(final_ens_yh_train > 0, 1, -1)
        return final_ens_yh_on_train_binarized
    # /Diversity Helper Functions

    # Diversity Calculations, initial models NOT aggregated
    def calculate_binary_diversity(self, all_columns, final_ensemble_column_train, ens_weights, binary=True):
        """ Calculates the diversity acoording to Chen&Bian, 2021
            Parameter:
                binary:     Must be True
            Returns diversity measure for each datapoint
        """
        n_samples = len(all_columns[0])
        # Margin of ensemble
        if binary:
            ens_column_on_train = self.binarize_ens_column_on_train(final_ensemble_column_train)
        else:
            ens_column_on_train = final_ensemble_column_train

        norm_factor = 0.5
        ens_column_on_train = ens_column_on_train * norm_factor

        # Create weigthed margins of ALL individual base learners
        weights_formatted = np.expand_dims(ens_weights, axis=[1, 2])
        all_weighted_margins = all_columns * weights_formatted

        sum_all_weighted_margins = np.sum(all_weighted_margins, axis=0) * norm_factor

        diversity = np.subtract(ens_column_on_train, sum_all_weighted_margins)
        avg_diversity = sum(diversity) / n_samples

        return avg_diversity, diversity
    # /Diversity Calculations

    # Helper Functions
    def calculate_average_bl_accuracy(self):
        self.average_bl_accuracy = np.average(self.bl_accuracies)
        return 0
    # /Helper Functions


if __name__ == '__main__':
    complete_return_names = ('initial_accuracy_on_train', 'initial_accuracy_on_val', 'ref_accuracy_on_train',
                             'ref_accuracy_on_val', 'extractor_output_shape', 'ens_accuracy_on_train',
                             'ens_accuracy_on_val', 'gen_time', 'ref_time', 'perf_time', 'gen_end')

    res_logger = ResultLogger()
    res_logger.update_attribute_map()
    res = res_logger.get_selected_results(complete_return_names)