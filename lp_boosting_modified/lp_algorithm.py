import numpy as np
import copy
import os
from problem.master_problem_base import MasterProblem
from problem.master_problem_alt import MasterProblemResiduals, MasterProblemRef, MasterProblemLPBoost, \
                                                  MasterProblemLPBoostActive, MasterProblemMissOnly
from problem.master_problem_mc_base import MasterProblemMultiClass
from problem.pricing_problem import PricingProblem, PricingProblemLinear
from ensemble.ensemble import Ensemble
from generation.generation import Bagging, EnhancedBagging

from utils.logger import DecolearnLogger
import logging
logger = logging.getLogger('soda')

# TODO: Calculate mp_value and div_datapoints in bagging phase too


class DecolearnAlgorithm:
    """ Description:
            Implements method calls of different parts of the algorithm
    """

    def __init__(self, exp_config: dict, res_logger, x_t_raw: np.ndarray,
                 y_t_raw: np.ndarray, x_v_raw: np.ndarray,
                 y_v_raw: np.ndarray, graph_boosting_iterations=False,
                 logflag=False, extra_iter=0,
                 logger_level='info', file_name=None, save_t=False):
        self._exp_config = exp_config
        self._n_samples = self._exp_config['n_sampl']
        self._extra_iter = extra_iter

        self.bagging = None
        self.mp = None
        self.mp_alt = None
        self._pp = None
        self._pp_alt = None

        # Data
        curr_dir = os.getcwd()
        main_dir = os.path.dirname(curr_dir)
        self.x_train, self.y_train, self.y_train_comp, self.x_val, self.y_val, \
            self.y_val_comp = self._format_data(x_t_raw[:self._n_samples],
                                                y_t_raw[:self._n_samples],
                                                x_v_raw, y_v_raw, self._exp_config['target_'])
        if save_t:
            np.save(main_dir + '/results/x_train', self.x_train)
            np.save(main_dir + '/results/y_train', self.y_train)
            np.save(main_dir + '/results/x_val', self.x_val)
            np.save(main_dir + '/results/y_val', self.y_val)
            np.save(main_dir + '/results/y_train_comp', self.y_train_comp)
            np.save(main_dir + '/results/y_val_comp', self.y_val_comp)

        self.y_raw = y_t_raw[:self._n_samples]

        self._model_id = 0

        # Instantiate ensemble object with or without y_train/y_val depending on binary_mode
        self.ens = Ensemble(self.x_train,  self.x_val, self.y_train_comp, self.y_val_comp,
                            self._exp_config['n_start'], self._exp_config['comb_me'])
        self.ens_alt = None

        self.show_boosting_iterations = graph_boosting_iterations
        self._logger_activated = logflag

        if self._exp_config['comb_me'] == 'vo':
            self._binarize_preds = True
        else:
            self._binarize_preds = False

        # Results
        self.res_logger = res_logger
        self.res_logger_alt = None

        # create logger handler
        self.logger_handler = DecolearnLogger(logger_level, file_name)

        self._check_input(self._exp_config['bl_config'], self._exp_config['bl_type'])
        self._check_input(self._exp_config['bl_config_ref'], self._exp_config['bl_type_ref'])

    @staticmethod
    def _check_input(bl_config, bl_type):
        len_bl_config = len(bl_config)
        if bl_type.lower() not in ['mlp', 'cnn', 'tree', 'xgboost', 'enscnnvgg']:
            raise ValueError(f'{bl_type} is not defined')

        if bl_type.lower() == 'mlp' and len_bl_config != 8:
            raise Exception(f'Expected arguments for MLP: 7, given: {len_bl_config}')

        if bl_type.lower() == 'cnn' and len_bl_config != 12:
            raise Exception(f'Expected arguments for CNN: 12, given: {len_bl_config}')

        if bl_type.lower() == 'xg_boost' and len_bl_config != 7:
            raise Exception(f'Expected argument for XGBoost: 7, given : {len_bl_config}')


    @staticmethod
    def _format_data(x_t, y_t, x_v, y_v, target):
        """
        - y_train_comp and y_val_comp have values 1 for multi-case
        :param x_t: Feature train data
        :param y_t: Label train data
        :param x_v: Feature validation data
        :param y_v: Label validation data
        :param target: Label target to be discriminiated agsint
        :return:
        :rtype:
        """
        x_train = x_t / 255
        x_val = x_v / 255

        y_train = np.asarray([1 if x == target else 0 for x in y_t], dtype=np.short)
        y_train = y_train.astype('float64')
        y_train_comp = np.asarray([1 if x == target else -1 for x in y_t], dtype=np.short)
        y_train_comp = y_train_comp.astype('float64')

        y_val_comp = np.asarray([1 if x == target else -1 for x in y_v], dtype=np.short)
        y_val_comp = y_val_comp.astype('float64')
        y_val = 1
        return x_train, y_train, y_train_comp, x_val, y_val, y_val_comp

    def generate(self, ebagging=False):
        """ Description:
                - To get eBagging performance history, call bagging_obj.get_bagging_perf()
        """
        _, initial_bl_config = zip(*self._exp_config['bl_config'].items())
        if ebagging:
            self.bagging = EnhancedBagging(self.x_train, self.y_train, self.y_train_comp, self.x_val,
                                           self.y_val, self.y_val_comp,
                                           self._exp_config['bl_type'], initial_bl_config,
                                           self._exp_config['n_start'], self._exp_config['comb_me'], self.ens,
                                           self._exp_config['binary_mode'],
                                           n_bags=self._exp_config['n_bags'],
                                           bootstrapping=self._exp_config['bootstrapping'],
                                           result_logger=self.res_logger)
        else:
            self.bagging = Bagging(self.x_train, self.y_train, self.y_train_comp, self.x_val,
                                   self.y_val, self.y_val_comp,
                                   self._exp_config['bl_type'], initial_bl_config,
                                   self._exp_config['n_start'], self._exp_config['comb_me'], self.ens,
                                   self._exp_config['binary_mode'],
                                   n_bags=self._exp_config['n_bags'],
                                   bootstrapping=self._exp_config['bootstrapping'],
                                   result_logger=self.res_logger)
        self.bagging.get_bl_preds_on_val()
        # self.bagging.get_bagging_perf()
        self.bagging.currently_activated = self._exp_config['n_start']
        self._model_id = self.bagging.model_id
        self.res_logger.gen_end = self._exp_config['n_start']

        # Forking after generation phase, Note: Own implementation of copying for ensemble obj.
        self.res_logger_alt = copy.deepcopy(self.res_logger)
        self.ens_alt = self.ens.__deepcopy__()

        return 0

    def refine(self, graph=False):
        _, refinement_bl_config = zip(*self._exp_config['bl_config_ref'].items())

        self._pp = PricingProblem(self.x_train, self.y_train, self._exp_config['bl_type_ref'],
                                  refinement_bl_config, self._model_id + 1)

        penalty = 1 / (len(self.x_train) * self._exp_config['nu'])
        self.mp = MasterProblem(penalty, self.ens, self._pp, self.x_train, self.x_val, self.y_train,
                                self.y_train_comp, self.y_val, self.y_val_comp, self._exp_config,
                                extra_iter=self._extra_iter, result_logger=self.res_logger)
        self.mp.refine(logger=self._logger_activated, graph=graph)

        # self.refine_alt(graph=graph)

        return 0

    def refine_alt(self, graph=False):
        # prev_iter_n = self.mp.iteration_counter - 1
        _, refinement_bl_config = zip(*self._exp_config['bl_config_ref'].items())
        penalty = 1 / (len(self.x_train) * self._exp_config['nu'])

        penalty = 1 / (len(self.x_train) * self._exp_config['nu'])
        _, refinement_bl_config = zip(*self._exp_config['bl_config_ref'].items())

        self._pp_lpboost = PricingProblemLinear(self.x_train, self.y_train_comp, self._exp_config['bl_type_ref'],
                                                refinement_bl_config, self._model_id + 1)
        self._mp_lpboost = MasterProblemLPBoost(penalty, self.ens_alt, self._pp_lpboost, self.x_train, self.y_train_comp,
                                                self.y_train_comp, self._exp_config, extra_iter=self._extra_iter,
                                                result_logger=self.res_logger_alt)
        self._mp_lpboost.refine(logger=self._logger_activated, graph=graph, var_gen_end=self._exp_config['n_start'])

    def final_results_on_train_bagging(self):
        """ Description:
                - Use only when initial models are aggregated (Bagging)
        """
        self.ens.generate_ens_fin_pred_on_train_bagging()
        return self.ens.final_ens_pred_on_train

    def final_results_on_val_bagging(self):
        """ Description:
                - Use only when initial models are aggregated (Bagging)
        """
        self.ens.generate_ens_fin_pred_on_val_bagging()
        return self.ens.final_ens_pred_on_val

    def compare_to_reweighted_initial_ens_on_train(self):
        """ Description:
                - Initial prediction scores if ensemble was built with weighted models.
                - The weights of the individual models are determined according to their performance on x_train.
                - The results are given for x_train.
        """

        model_i_miss_score = []
        for indi_model_yh in self.ens.initial_columns_on_train:
            score = 0
            for yh in indi_model_yh:
                if yh <= 0:
                    score += 1
            model_i_miss_score.append(score)

        sum_miss = sum(model_i_miss_score)
        pre_weights = []
        for miss_score in model_i_miss_score:
            if miss_score == 0:
                print('compare_to_reweighted_initial_ens_on_train: Miss score is zero. Terminate algorithm')
                quit()
            pre_weights.append(sum_miss / miss_score)

        sum_pre_weights = sum(pre_weights)
        weights = np.asarray(pre_weights) / sum_pre_weights

        initial_preds = np.asarray(self.ens.initial_preds_on_train)
        if self._exp_config['comb_me'] == 'av':
            improved_initial_ens_on_train = np.zeros(len(self.y_train))

            for idx, preds in enumerate(initial_preds):
                improved_initial_ens_on_train += (preds * weights[idx])

        else:
            zeros = []
            for index, weight in enumerate(weights):
                if weight < 0.1:
                    zeros.append(index)
            all_valid_preds = np.delete(initial_preds, zeros, axis=0)
            improved_initial_ens_on_train = np.array([1 if x > 0 else -1 for x in np.sum(all_valid_preds, axis=0)])

        improved_initial_ens_on_train_yh = improved_initial_ens_on_train * self.y_train_comp
        miss_score_ens = 0

        for yh in improved_initial_ens_on_train_yh:
            if yh <= 0:
                miss_score_ens += 1

        print(f"The re-weighted initial ensemble miss score on x_train is:        {miss_score_ens}")

        return improved_initial_ens_on_train

    def compare_initial_final_ens_on_both(self):
        """ Description:
                Compare initial ensemble after gen. phase and final ensemble predictions on
                both x_train and x_val
        """

        # Alternative mp ens miss score on x_train/x_val
        miss_ini_ens_train = self.res_logger.miss_mp_initial_ens_train   # Unlist
        miss_ini_ens_val = self.res_logger.miss_mp_initial_ens_val

        # Final ensemble on x_train/x_val
        fin_miss_occ_train = np.where(self.ens.final_ens_column_on_train <= 0, 1, 0)
        miss_fin_ens_train, = sum(fin_miss_occ_train)
        fin_miss_occ_val = np.where(self.ens.final_ens_column_on_val <= 0, 1, 0)
        miss_fin_ens_val, = sum(fin_miss_occ_val)

        # Initial with Bagging
        initial_miss_occ_train_bagging = np.where(self.ens.initial_bagging_ens_column_on_train <= 0, 1, 0)
        miss_ini_ens_train_bagging, = sum(initial_miss_occ_train_bagging)

        initial_miss_occ_val_bagging = np.where(self.ens.initial_bagging_ens_column_on_val <= 0, 1, 0)
        miss_ini_ens_val_bagging, = sum(initial_miss_occ_val_bagging)

        print('\n')
        print(f'The initial BAGGING miss score on x_train is:                   {miss_ini_ens_train_bagging}')
        print(f'The initial BAGGING miss score on x_val is:                     {miss_ini_ens_val_bagging}')
        print('\n')
        print(f'The initial MP-calculated ensemble miss score on x_train is:   {miss_ini_ens_train}')
        print(f'The initial MP-calculated ensemble miss score on x_val is:     {miss_ini_ens_val}')
        print('\n')
        print(f'The final ensemble miss score on x_train is:                    {miss_fin_ens_train} ')
        print(f'The final ensemble miss score on x_val is:                      {miss_fin_ens_val}\n')

    def compare_initial_final_ens_on_both_alt(self):
        """ Compare initial ensemble after gen. phase and final ensemble predictions on
        both x_train and x_val
        """

        # Alternative mp ens miss score on x_train/x_val
        miss_ini_ens_train = self.res_logger_alt.miss_mp_initial_ens_train  # Unlist
        miss_ini_ens_val = self.res_logger_alt.miss_mp_initial_ens_val

        # Final ensemble on x_train/x_val
        fin_miss_occ_train = np.where(self.ens_alt.final_ens_column_on_train <= 0, 1, 0)
        miss_fin_ens_train, = sum(fin_miss_occ_train)
        fin_miss_occ_val = np.where(self.ens_alt.final_ens_column_on_val <= 0, 1, 0)
        miss_fin_ens_val, = sum(fin_miss_occ_val)

        # Initial with Bagging
        initial_miss_occ_train_bagging = np.where(self.ens_alt.initial_bagging_ens_column_on_train <= 0, 1, 0)
        miss_ini_ens_train_bagging, = sum(initial_miss_occ_train_bagging)

        initial_miss_occ_val_bagging = np.where(self.ens_alt.initial_bagging_ens_column_on_val <= 0, 1, 0)
        miss_ini_ens_val_bagging, = sum(initial_miss_occ_val_bagging)

        print('\n')
        print(f'The initial BAGGING miss score on x_train is:                   {miss_ini_ens_train_bagging}')
        print(f'The initial BAGGING miss score on x_val is:                     {miss_ini_ens_val_bagging}')
        print('\n')
        print(f'The alternative initial MP ensemble miss score on x_train is:   {miss_ini_ens_train}')
        print(f'The alternative initial MP ensemble miss score on x_val is:     {miss_ini_ens_val}')
        print('\n')
        print(f'The final ensemble miss score on x_train is:                    {miss_fin_ens_train} ')
        print(f'The final ensemble miss score on x_val is:                      {miss_fin_ens_val}\n')


class DecolearnAlgorithmDataReductionFocused(DecolearnAlgorithm):
    def __init__(self, exp_config: dict, res_logger, x_t_raw: np.ndarray, y_t_raw: np.ndarray, x_v_raw: np.ndarray,
                 y_v_raw: np.ndarray, graph_boosting_iterations=False, logflag=False, extra_iter=0,
                 logger_level='info', file_name=None, save_t=False):
        self.gen_end = 0
        self.min_change_threshold = 0.2     # Old: 0.2
        self.last_gen_u = None
        super().__init__(exp_config, res_logger, x_t_raw, y_t_raw, x_v_raw, y_v_raw, graph_boosting_iterations,
                         logflag=logflag, extra_iter=extra_iter, logger_level=logger_level, file_name=file_name,
                         save_t=save_t)

    def generate(self, graph_gen=False, ebagging=False):
        """ Description:
                - To get eBagging performance history, call bagging_obj.get_bagging_perf()
        """
        # dual_miss_threshold = 0.0
        min_change = True
        n_bags = self._exp_config['n_bags']
        gen_iter = 0

        _, initial_bl_config = zip(*self._exp_config['bl_config'].items())
        if ebagging:
            self.bagging = EnhancedBagging(self.x_train, self.y_train, self.y_train_comp, self.x_val, self.y_val,
                                           self.y_val_comp, self._exp_config['bl_type'], initial_bl_config,
                                           self._exp_config['n_start'], self._exp_config['comb_me'], self.ens,
                                           self._exp_config['binary_mode'], n_bags=n_bags, bootstrapping=True,
                                           result_logger=self.res_logger)
        else:
            self.bagging = Bagging(self.x_train, self.y_train, self.y_train_comp, self.x_val, self.y_val,
                                   self.y_val_comp, self._exp_config['bl_type'], initial_bl_config,
                                   self._exp_config['n_start'], self._exp_config['comb_me'], self.ens,
                                   self._exp_config['binary_mode'], n_bags=n_bags, bootstrapping=True,
                                   result_logger=self.res_logger)

        for idx, batch in enumerate(self.bagging.newly_inner_loop):
            gen_iter += 1
            print(f'########################## GEN Iteration {gen_iter} ##########################')
            current_ini_activated = [idx+1]
            pred, column, model = batch
            self.ens.add_initial_pred_on_train(pred)
            self.ens.add_initial_column_on_train(column)
            self.ens.add_initial_H(model)
            self.res_logger.add_margin('Not_Logged')
            # Method adds val preds automatically to ensemble
            self.bagging.get_bl_preds_on_val(idx)
            # Is only used to log the first model data
            self.bagging.get_bagging_perf(with_active=current_ini_activated)

        self._model_id = self.bagging.model_id

        logger.info('Construct pricing problem for bagging')
        _, refinement_bl_config = zip(*self._exp_config['bl_config_ref'].items())
        self._pp = PricingProblem(self.x_train, self.y_train, self._exp_config['bl_type_ref'],
                                  refinement_bl_config, self._model_id + 1)
        penalty = 1 / (len(self.x_train) * self._exp_config['nu'])
        self.mp = MasterProblem(penalty, self.ens, self._pp, self.x_train, self.x_val, self.y_train,
                                self.y_train_comp, self.y_val, self.y_val_comp, self._exp_config, extra_iter=self._extra_iter,
                                result_logger=self.res_logger)

        u = self.mp.solve_in_dual_for_gen()
        self.res_logger.add_u_i(u)
        self.res_logger.add_mp_val_i(self.mp.beta * (-1))

        len_u = len(u)
        last_u = np.copy(u)

        # while min_change:
        while gen_iter <= 5:
            self.bagging.add_bls(1)
            # Calculate MP and log results for each newly added BL trained on bags
            for idx, batch in enumerate(self.bagging.newly_inner_loop):
                gen_iter += 1
                print(f'########################## GEN Iteration {gen_iter} ##########################')

                pred, column, model = batch
                self.ens.add_initial_pred_on_train(pred)
                self.ens.add_initial_column_on_train(column)
                self.ens.add_initial_H(model)
                u = self.mp.solve_in_dual_for_gen()
                self.res_logger.add_u_i(u)
                self.res_logger.add_margin('Not_Logged')
                self.bagging.get_bl_preds_on_val(idx)
                # TODO: Fix: Alpha vector solely comes from mp.log_progress()
                if self._logger_activated:
                    self.mp.log_progress(in_boosting=False)
                    if graph_gen:
                        self.mp.graph_progress()

            truth_table = (u == last_u)
            number_unchanged = sum(truth_table)
            diff_score = len_u - number_unchanged
            if (diff_score / len_u) < self.min_change_threshold:
                min_change = False
            # min_change = not all(u == last_u)
            last_u = np.copy(u)
            print(f'## Number of active datapoints changed in iteration {gen_iter}: {diff_score} ###')
        # self.gen_end = gen_iter - 1
        # This counts the initial model as the first bagging iteration in the XML file
        self.last_gen_u = u
        self.gen_end = gen_iter
        self.res_logger.gen_end = self.gen_end
        self.mp.iteration_counter = 0

        # For forking after generation phase, Note: Own implementation of copying for ensemble obj.
        self.res_logger_alt = copy.deepcopy(self.res_logger)
        self.ens_alt = self.ens.__deepcopy__()

        return 0

    def refine(self, graph=False):
        self.mp.refine(logger=self._logger_activated, graph=graph, var_gen_end=self.gen_end)
        # self.refine_alt(graph=graph)

        return 0

    def refine_alt(self, graph=False):
        prev_iter_n = self.mp.iteration_counter - 1
        _, refinement_bl_config = zip(*self._exp_config['bl_config_ref'].items())
        penalty = 1 / (len(self.x_train) * self._exp_config['nu'])

        # Instantiate/copy alternative containers for alternative MP solution
        self._pp_alt = PricingProblemLinear(self.x_train, self.y_train, self._exp_config['bl_type_ref'],
                                            refinement_bl_config, self._model_id + 1)

        self.mp_alt = MasterProblemLPBoost(penalty, self.ens_alt, self._pp_alt, self.x_train, self.y_train,
                                           self.y_train_comp, self.y_val, self.y_val_comp, self._exp_config,
                                           extra_iter=self._extra_iter, result_logger=self.res_logger_alt)
        self.mp_alt._u = self.last_gen_u
        self.mp_alt.refine(logger=self._logger_activated, graph=graph, var_gen_end=self.gen_end)


class DecolearnAlgorithmLPBoost(DecolearnAlgorithmDataReductionFocused):
    def __init__(self, exp_config: dict, res_logger, x_t_raw: np.ndarray, y_t_raw: np.ndarray, x_v_raw: np.ndarray,
                 y_v_raw: np.ndarray, graph_boosting_iterations=False, logflag=False, extra_iter=0,
                 logger_level='debug', file_name=None, save_t=False):
        self.gen_end = 0
        self.min_change_threshold = 0.2
        self._pp_lpboost = None
        self._mp_lpboost = None
        super().__init__(exp_config, res_logger, x_t_raw, y_t_raw, x_v_raw, y_v_raw, graph_boosting_iterations,
                         logflag=logflag, extra_iter=extra_iter, logger_level=logger_level, file_name=file_name,
                         save_t=save_t)

    def refine(self, graph=False):
        penalty = 1 / (len(self.x_train) * self._exp_config['nu'])
        _, refinement_bl_config = zip(*self._exp_config['bl_config_ref'].items())

        self._pp_lpboost = PricingProblemLinear(self.x_train, self.y_train_comp, self._exp_config['bl_type_ref'],
                                                refinement_bl_config, self._model_id + 1)
        self._mp_lpboost = MasterProblemLPBoost(penalty, self.ens, self._pp_lpboost, self.x_train, self.x_val,
                                                self.y_train_comp, self.y_train_comp, self._exp_config, extra_iter=self._extra_iter,
                                                result_logger=self.res_logger)
        self._mp_lpboost.refine(logger=self._logger_activated, graph=graph, var_gen_end=self.gen_end)
        # self.refine_alt(graph=graph)

        return 0

    def refine_alt(self, graph=False):
        _, refinement_bl_config = zip(*self._exp_config['bl_config_ref'].items())
        penalty = 1 / (len(self.x_train) * self._exp_config['nu'])

        self._pp_alt = PricingProblem(self.x_train, self.y_train, self._exp_config['bl_type_ref'],
                                      refinement_bl_config, self._model_id + 1)

        self.mp_alt = MasterProblem(penalty, self.ens_alt, self._pp_alt, self.x_train, self.y_train,
                                    self.y_train_comp, self._exp_config, extra_iter=self._extra_iter,
                                    result_logger=self.res_logger_alt)
        self.mp_alt._u = self.last_gen_u
        self.mp_alt.refine(logger=self._logger_activated, graph=graph, var_gen_end=self.gen_end)


class DecolearnAlgorithmLPBoostActiveData(DecolearnAlgorithmDataReductionFocused):
    def __init__(self, exp_config: dict, res_logger, x_t_raw: np.ndarray, y_t_raw: np.ndarray, x_v_raw: np.ndarray,
                 y_v_raw: np.ndarray, graph_boosting_iterations=False, logflag=False, extra_iter=0, save_t=False):
        self.gen_end = 0
        self.min_change_threshold = 0.2
        super().__init__(exp_config, res_logger, x_t_raw, y_t_raw, x_v_raw, y_v_raw, graph_boosting_iterations,
                         logflag=logflag, extra_iter=extra_iter, save_t=save_t)

    def refine(self, graph=False):
        penalty = 1 / (len(self.x_train) * self._exp_config['nu'])
        _, refinement_bl_config = zip(*self._exp_config['bl_config_ref'].items())

        self._pp = PricingProblemLinear(self.x_train, self.y_train_comp, self._exp_config['bl_type_ref'],
                                        refinement_bl_config, self._model_id + 1)
        self.mp = MasterProblemLPBoostActive(penalty, self.ens, self._pp, self.x_train, self.y_train_comp,
                                            self.y_train_comp, self._exp_config, extra_iter=self._extra_iter,
                                            result_logger=self.res_logger)
        self.mp.refine(logger=self._logger_activated, graph=graph, var_gen_end=self.gen_end)
        # self.refine_alt(graph=graph)

        return 0

    def refine_alt(self, graph=False):
        _, refinement_bl_config = zip(*self._exp_config['bl_config_ref'].items())
        penalty = 1 / (len(self.x_train) * self._exp_config['nu'])

        # Instantiate/copy alternative containers for alternative MP solution
        self._pp_alt = PricingProblem(self.x_train, self.y_train, self._exp_config['bl_type_ref'],
                                      refinement_bl_config, self._model_id + 1)
        self.mp_alt = MasterProblem(penalty, self.ens_alt, self._pp_alt, self.x_train, self.y_train,
                                    self.y_train_comp, self._exp_config, extra_iter=self._extra_iter,
                                    result_logger=self.res_logger_alt)
        self.mp_alt.refine(logger=self._logger_activated, graph=graph, var_gen_end=self.gen_end)


class DecolearnAlgorithmMissOnly(DecolearnAlgorithmDataReductionFocused):
    def __init__(self, exp_config: dict, res_logger, x_t_raw: np.ndarray, y_t_raw: np.ndarray, x_v_raw: np.ndarray,
                 y_v_raw: np.ndarray, graph_boosting_iterations=False, logflag=False, extra_iter=0,
                 logger_level='debug', file_name=None, save_t=False):
        self.gen_end = 0
        self.min_change_threshold = 0.2
        self._pp = None
        self._mp = None
        super().__init__(exp_config, res_logger, x_t_raw, y_t_raw, x_v_raw, y_v_raw, graph_boosting_iterations,
                         logflag=logflag, extra_iter=extra_iter, logger_level=logger_level, file_name=file_name,
                         save_t=save_t)

    def refine(self, graph=False):
        penalty = 1 / (len(self.x_train) * self._exp_config['nu'])
        _, refinement_bl_config = zip(*self._exp_config['bl_config_ref'].items())

        self._pp = PricingProblem(self.x_train, self.y_train, self._exp_config['bl_type_ref'],
                                  refinement_bl_config, self._model_id + 1)
        self._mp = MasterProblemMissOnly(penalty, self.ens, self._pp, self.x_train, self.y_train,
                                         self.y_train_comp, self._exp_config, extra_iter=self._extra_iter,
                                         result_logger=self.res_logger)
        self._mp._u = self.last_gen_u
        self._mp.refine(logger=self._logger_activated, graph=graph, var_gen_end=self.gen_end)
        # self.refine_alt(graph=graph)

        return 0

    def refine_alt(self, graph=False):
        _, refinement_bl_config = zip(*self._exp_config['bl_config_ref'].items())
        penalty = 1 / (len(self.x_train) * self._exp_config['nu'])

        self._pp_alt = PricingProblem(self.x_train, self.y_train, self._exp_config['bl_type_ref'],
                                      refinement_bl_config, self._model_id + 1)

        self.mp_alt = MasterProblem(penalty, self.ens_alt, self._pp_alt, self.x_train, self.y_train,
                                    self.y_train_comp, self._exp_config, extra_iter=self._extra_iter,
                                    result_logger=self.res_logger_alt)
        self.mp_alt._u = self.last_gen_u
        self.mp_alt.refine(logger=self._logger_activated, graph=graph, var_gen_end=self.gen_end)


"""
############################
### Multi-class versions ###
############################
"""


class DecolearnAlgorithmMulti:
    def __init__(self, exp_config: dict, res_logger, x_t_raw: np.ndarray, y_t_raw: np.ndarray, x_v_raw: np.ndarray,
                 y_v_raw: np.ndarray, graph_boosting_iterations=False, logflag=False, logger_level='info',
                 file_name=None, extra_iter=0, save_t=False):
        """
        :param exp_config: Experiment configuration as dictionary
        :param res_logger: Instance of result logger given from run file
        :param x_t_raw: Raw data of train features
        :param y_t_raw: Raw data of train labels
        :param x_v_raw: Raw data of validation features
        :param y_v_raw: Raw data of validation labels
        :param graph_boosting_iterations: Graph the accuracies in each iteration
        :param logflag: Indicates if logging occurs
        :param file_name: Save XML-file under file_name
        :param extra_iter: Additional iterations after termination condition is reached
        :param save_t: Indicates of train data is saved
        """
        self.binary_mode = exp_config['binary_mode']
        self._exp_config = exp_config
        self._n_samples = self._exp_config['n_sampl']
        self._extra_iter = extra_iter

        self.bagging = None
        self.mp = None
        self._pp = None

        # Input Interception
        if self._exp_config['comb_me'] != 'av':
            raise ValueError('Only permissible combination mechanism in multi-mode: "av"')

        # Data
        curr_dir = os.getcwd()
        main_dir = os.path.dirname(curr_dir)

        self.x_train, self.y_train, self.y_train_comp, self.x_val, self.y_val, \
            self.y_val_comp = self._format_data(x_t_raw[:self._n_samples],
                                                y_t_raw[:self._n_samples],
                                                x_v_raw, y_v_raw)
        # Instantiate ensemble object with  y_train and y_val
        self.container = Ensemble(self.x_train, self.x_val, self.y_train_comp, self.y_val_comp,
                                  self._exp_config['n_start'], self._exp_config['comb_me'], y_train=self.y_train,
                                  y_val=self.y_val, binary_mode=False)

        if save_t:
            np.save(main_dir + '/results/x_train', self.x_train)
            np.save(main_dir + '/results/y_train', self.y_train)
            np.save(main_dir + '/results/x_val', self.x_val)
            np.save(main_dir + '/results/y_val', self.y_val)
            np.save(main_dir + '/results/y_train_comp', self.y_train_comp)
            np.save(main_dir + '/results/y_val_comp', self.y_val_comp)

        self._model_id = 0

        # # Instantiate ensemble object with  y_train and y_val
        # self.container = Ensemble(self.x_train, self.x_val, self.y_train_comp, self.y_val_comp,
        #                           self._exp_config['n_start'], self._exp_config['comb_me'], y_train=self.y_train,
        #                           y_val=self.y_val, binary_mode=False)

        self.show_boosting_iterations = graph_boosting_iterations
        self._logger_activated = logflag

        # Results
        self.res_logger = res_logger

        # create logger handler
        self.logger_handler = DecolearnLogger(logger_level, file_name)

    @property
    def exp_config(self):
        return self._exp_config

    @staticmethod
    def check_bl_config(bl_config, bl_type):
        len_bl_config = len(bl_config)
        if bl_type.lower() not in ['mlp', 'cnn', 'tree', 'xgboost']:
            raise ValueError(f'{bl_type} is not defined')

        if bl_type.lower() == 'mlp' and len_bl_config != 8:
            raise Exception(f'Expected arguments for MLP: 7, given: {len_bl_config}')

        if bl_type.lower() == 'cnn' and len_bl_config != 12:
            raise Exception(f'Expected arguments for CNN: 12, given: {len_bl_config}')

        if bl_type.lower() == 'xg_boost' and len_bl_config != 7:
            raise Exception(f'Expected argument for XGBoost: 7, given : {len_bl_config}')

    @staticmethod
    def _format_data(x_t, y_t, x_v, y_v):
        """ Description:
                For interface standardization, include y_train_comp, y_val_comp
                x_raw and x_val: Normalized
                y_train and y_val: Returned as is

        """
        x_train = x_t / 255
        x_val = x_v / 255

        y_train = y_t
        y_train_comp = np.asarray([1])
        y_val_comp = np.asarray([1])
        y_val = y_v

        return x_train, y_train, y_train_comp, x_val, y_val, y_val_comp

    def generate(self, ebagging=False):
        """ Description:
                - To get eBagging performance history, call bagging_obj.get_bagging_perf()
        """
        _, initial_bl_config = zip(*self._exp_config['bl_config'].items())
        gen_iter = 0

        if ebagging:
            self.bagging = EnhancedBagging(self.x_train, self.y_train, self.y_train_comp, self.x_val,
                                           self.y_val, self.y_val_comp,
                                           self._exp_config['bl_type'], initial_bl_config,
                                           self._exp_config['n_start'], self._exp_config['comb_me'], self.container,
                                           self._exp_config['binary_mode'], n_bags=self._exp_config['n_bags'],
                                           bootstrapping=False, result_logger=self.res_logger)

        else:
            self.bagging = Bagging(self.x_train, self.y_train, self.y_train_comp, self.x_val,
                                   self.y_val, self.y_val_comp,
                                   self._exp_config['bl_type'], initial_bl_config,
                                   self._exp_config['n_start'], self._exp_config['comb_me'], self.container,
                                   self._exp_config['binary_mode'], n_bags=self._exp_config['n_bags'],
                                   bootstrapping=False, result_logger=self.res_logger)

        # self.bagging.currently_activated = self._exp_config['n_start']
        # self.bagging.get_bl_preds_on_val()
        # # Is only used to log the first model data, n=1, weights=1
        # self.bagging.get_bagging_perf()

        for idx, batch in enumerate(self.bagging.newly_inner_loop):
            gen_iter += 1
            print(f'########################## GEN Iteration {gen_iter} ##########################')
            current_ini_activated = [idx+1]
            pred, column, model = batch
            self.container.add_initial_pred_on_train(pred)
            self.container.add_initial_column_on_train(column)
            self.container.add_initial_H(model)
            self.res_logger.add_margin('Not_Logged')
            self.res_logger.add_u_i('Initial_Phase')
            self.bagging.get_bl_preds_on_val(idx)
            # Is only used to log the first model data
            self.bagging.get_bagging_perf(with_active=current_ini_activated)

        self._model_id = self.bagging.model_id
        self.res_logger.gen_end = gen_iter

        return 0

    def refine(self, graph=False):
        _, refinement_bl_config = zip(*self._exp_config['bl_config_ref'].items())

        self._pp = PricingProblem(self.x_train, self.y_train, self._exp_config['bl_type_ref'],
                                  refinement_bl_config, self._model_id + 1)

        # Under the assumption that all \lambda_{i,m} assume the maximal value: len(np.unique(self.y_train))
        penalty = 1 / (len(self.x_train) * self._exp_config['nu'])
        self.mp = MasterProblemMultiClass(penalty, self.container, self._pp, self.x_train, self.y_train, self.y_val,
                                          self._exp_config, extra_iter=self._extra_iter, result_logger=self.res_logger)
        self.mp.refine(logger=self._logger_activated)

        # self.refine_alt(graph=graph)

        return 0

    def generate_final_results_on_train(self):
        """ Description:
                - Use only when initial models are NOT aggregated
        """
        self.container.generate_ens_fin_pred_on_train_multi()
        return self.container.final_ens_pred_on_train

    def generate_final_results_on_val(self):
        """ Description:
                - Use only when initial models are NOT aggregated
        """
        self.container.generate_ens_fin_pred_on_val_multi()
        return self.container.final_ens_pred_on_val


class DecolearnAlgorithmMultiDataReductionFocused(DecolearnAlgorithmMulti):
    def __init__(self, exp_config: dict, res_logger, x_t_raw: np.ndarray, y_t_raw: np.ndarray, x_v_raw: np.ndarray,
                 y_v_raw: np.ndarray, graph_boosting_iterations=False, logflag=False, logger_level='info',
                 file_name=None, extra_iter=0, save_t=False):
        super().__init__(exp_config, res_logger, x_t_raw, y_t_raw, x_v_raw, y_v_raw,
                         graph_boosting_iterations=graph_boosting_iterations,
                         logflag=logflag, logger_level=logger_level, file_name=file_name, extra_iter=extra_iter,
                         save_t=save_t)

    def generate(self, ebagging=False):
        """
        :param ebagging: Switcher between normal bagging and ebagging
        :return:
        """
        _, initial_bl_config = zip(*self._exp_config['bl_config'].items())
        gen_iter = 0

        if ebagging:
            self.bagging = EnhancedBagging(self.x_train, self.y_train, self.y_train_comp, self.x_val,
                                           self.y_val, self.y_val_comp,
                                           self._exp_config['bl_type'], initial_bl_config,
                                           1, self._exp_config['comb_me'], self.container,
                                           self._exp_config['binary_mode'], n_bags=1,
                                           bootstrapping=False, result_logger=self.res_logger)

        else:
            self.bagging = Bagging(self.x_train, self.y_train, self.y_train_comp, self.x_val,
                                   self.y_val, self.y_val_comp,
                                   self._exp_config['bl_type'], initial_bl_config,
                                   1, self._exp_config['comb_me'], self.container,
                                   self._exp_config['binary_mode'], n_bags=1,
                                   bootstrapping=False, result_logger=self.res_logger)

        # self.bagging.currently_activated = self._exp_config['n_start']
        # self.bagging.get_bl_preds_on_val()
        # # Is only used to log the first model data, n=1, weights=1
        # self.bagging.get_bagging_perf()

        for idx, batch in enumerate(self.bagging.newly_inner_loop):
            gen_iter += 1
            print(f'########################## BL Bagging  {gen_iter} ##########################')
            current_ini_activated = [idx + 1]
            pred, column, model = batch
            self.container.add_initial_pred_on_train(pred)
            self.container.add_initial_column_on_train(column)
            self.container.add_initial_H(model)
            self.res_logger.add_margin('Not_Logged')
            # self.res_logger.add_u_i('Initial_Phase')
            self.bagging.get_bl_preds_on_val(idx)
            # Is only used to log the first model data
            self.bagging.get_bagging_perf(with_active=current_ini_activated)

        self._model_id = self.bagging.model_id

        _, refinement_bl_config = zip(*self._exp_config['bl_config_ref'].items())

        # Instantaite MP and PP
        self._pp = PricingProblem(self.x_train, self.y_train, self._exp_config['bl_type_ref'],
                                  refinement_bl_config, self._model_id + 1)

        penalty = 1 / (len(self.x_train) * self._exp_config['nu'])
        self.mp = MasterProblemMultiClass(penalty, self.container, self._pp, self.x_train, self.y_train, self.y_val,
                                          self._exp_config, extra_iter=self._extra_iter, result_logger=self.res_logger)

        u = self.mp.solve_in_dual_for_gen()
        self.res_logger.add_u_i(u)

        while gen_iter < self._exp_config['n_start']:
            gen_iter += 1
            self.bagging.add_bls(1)
            # Calculate MP and log results for each newly added BL trained on bags
            for idx, batch in enumerate(self.bagging.newly_inner_loop):
                print(f'########### Convex Combination, Generation Iter:  {gen_iter} ###########')
                pred, column, model = batch
                self.container.add_initial_pred_on_train(pred)
                self.container.add_initial_column_on_train(column)
                self.container.add_initial_H(model)
                u = self.mp.solve_in_dual_for_gen()
                self.res_logger.add_u_i(u)
                self.res_logger.add_margin('Not_Logged')
                self.bagging.get_bl_preds_on_val(idx)
                # Alpha vector solely comes from mp.log_progress()
                if self._logger_activated:
                    self.mp.log_progress(in_boosting=False)

        # This counts the initial model as the first bagging iteration in the XML file
        self.mp.iteration_counter = 0

        self.res_logger.gen_end = gen_iter

        return 0

    def refine(self, graph=False):

        self.mp.refine(logger=self._logger_activated)

        # self.refine_alt(graph=graph)

        return 0