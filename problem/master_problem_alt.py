from __future__ import annotations
import numpy as np
import logging
from gurobipy import *
from typing import TYPE_CHECKING
from utils.grapher import Grapher
from problem.master_problem_base import MasterProblem
if TYPE_CHECKING:
    from ensemble.ensemble import Ensemble


class MasterProblemResiduals(MasterProblem):
    def __init__(self, d, ens_obj: Ensemble, pp, x_train, y_train, y_raw, y_train_comp, exp_config, target,
                 extra_iter=0, result_logger=None):
        self.y_raw = y_raw
        self.target = target
        self.miss_idx = None
        super().__init__(d, ens_obj, pp, x_train, y_train, y_train_comp, exp_config, extra_iter=extra_iter,
                         result_logger=result_logger)

    def solve_pp(self):
        baseline = self.ens.final_ens_pred_on_train
        domain = np.asarray(self.miss_idx)
        ref_model = self._pp.solve_pp_data_res(domain, baseline)
        ref_preds = (ref_model.get_r_prediction(self.x_train, batch_size=None))
        ref_yh = ref_preds * self.y_train_comp
        uyh = ref_yh * self._u
        sum_uyh = sum(uyh)

        return ref_model, ref_preds, ref_yh, sum_uyh

    def solve_in_dual(self, logger=True, graph=False):
        # PP is initialized with all data points, explicit dual weight formulation not needed, just vector length
        self._u = np.zeros((self._n_samples, 1))

        self.iterate_dual()

        # Prevents forkings in ensemble method. In first iteration, self.ens.final_ens... equals the first mp ens
        # alphas = self.solve_in_primal()
        alphas = self._m_d.getAttr("Pi")[:-1]
        alphas = [-i for i in alphas]
        self.ens.weights = alphas
        _, _ = self.ens.get_scores()

        self.ens.set_initial_mp_ens_pred_on_train(self.ens.final_ens_pred_on_train)
        self.ens.set_initial_mp_ens_column_on_train(self.ens.final_ens_column_on_train)
        self.res_log.calculate_miss_mp_initial_ens_train(self.ens.final_ens_column_on_train)

        self.ens.set_initial_mp_ens_pred_on_val(self.ens.final_ens_pred_on_val)
        self.ens.set_initial_mp_ens_column_on_val(self.ens.final_ens_column_on_val)
        self.res_log.calculate_miss_mp_initial_ens_val(self.ens.final_ens_column_on_val)
        # /Prevents forking in ensemble method

        self.miss_idx = np.where(self.ens.final_ens_column_on_train <= 0)[0]
        self._pp.update(self.x_train[self.miss_idx], self.y_train[self.miss_idx])
        ref_model, ref_preds, ref_yh, sum_uyh = self.solve_pp()

        iter = 0
        extra_iter_counter = 0
        while (sum_uyh > (self.beta + 1e-3) and iter < self._max_iter) or extra_iter_counter < self._extra_iter:
            if sum_uyh < (self.beta + 1e-3):
                extra_iter_counter += 1
                print(f'extra_iter_counter = {extra_iter_counter}')

            self.ens.add_refinement_H(ref_model)
            self.ens.add_refinement_pred_on_train(ref_preds)
            self.ens.add_refinement_column_on_train(ref_yh)

            self.iterate_dual()
            if logger:
                self.log_progress()

            self.miss_idx = np.where(self.ens.final_ens_column_on_train <= 0)[0]
            if len(self.miss_idx) == 0:
                break
            self._pp.update(self.x_train[self.miss_idx], self.y_train[self.miss_idx])
            ref_model, ref_preds, ref_yh, sum_uyh = self.solve_pp()
            print(f'Sum of yh is {sum_uyh}')
            print(f'BETA IS: {self.beta}')
            iter += 1

        if logger and graph:
            x = self.exp_config['n_start'] - 1
            single_data = ((x, self.res_log.miss_mp_initial_ens_train, 'mp ens. train'),
                           (x, self.res_log.miss_mp_initial_ens_val, 'mp ens. val'))
            grapher = Grapher()
            grapher.make_curve(title=f'Miss-classification Scores vs. Number of BLs --' +
                                     f'({self.exp_config["bl_type"]}-{self.exp_config["bl_type_ref"]})'
                               , label='L_m', x_label='Number of BLs',
                               y_label='Miss-classification Score of the Ensemble', color='black',
                               y=self.res_log.miss_score_train_history,
                               additional_ys=[self.res_log.miss_score_val_history],
                               additional_label=['L_m*'], marker='o', single_point_data=single_data,
                               vline=self.exp_config['n_start']-1)

            grapher.make_curve(title='Master Problem Value History', label='L_h',
                               x_label='Refinement Iteration',
                               y_label='Master Problem Value', color='blue', y=self.res_log.mp_history, marker='o')

            grapher.make_curve(title='Average Diversity of Data Points History', label='Avg. Div.',
                               x_label='Iteration', y_label='Diversity Measure', color='green',
                               y=self.res_log.avg_diversity_history, marker='o', vline=self.exp_config['n_start']-1)

            # for idx, diversity in enumerate(self.res_log.diversity_history):
            #     grapher.make_curve(title=f'Diversity of Data Points for Refinement Iteration: {idx}',
            #                        label=None, x_label='Datapoint', y_label='Diversity Measure',
            #                        color='green', y=diversity, marker=None, vline=self.exp_config['n_start']-1)

            grapher.show_plots()

            return 0


class MasterProblemRef(MasterProblem):
    """" Gives all train data to pricing problem"""
    def __init__(self, prev_iter, d, ens_obj: Ensemble, pp, x_train, y_train, y_train_comp, exp_config, extra_iter=0,
                 result_logger=None):
        self.prev_iter_n = prev_iter
        super().__init__(d, ens_obj, pp, x_train, y_train, y_train_comp, exp_config, extra_iter=extra_iter,
                         result_logger=result_logger)
        print('##############MP alt. initiated ####################')

    def solve_in_dual(self, logger=True, graph=False):
        # PP is initialized with all data points, explicit dual weight formulation not needed, just vector length
        self._u = np.zeros((self._n_samples, 1))

        self.iterate_dual()

        # Prevents forkings in ensemble method. In first iteration, self.ens.final_ens... equals the first mp ens
        # alphas = self.solve_in_primal()
        alphas = self._m_d.getAttr("Pi")[:-1]
        alphas = [-i for i in alphas]
        self.ens.weights = alphas
        # _, _ = self.ens.get_scores()
        self.ens.get_scores()

        self.ens.set_initial_mp_ens_pred_on_train(self.ens.final_ens_pred_on_train)
        self.ens.set_initial_mp_ens_column_on_train(self.ens.final_ens_column_on_train)
        self.res_log.calculate_miss_mp_initial_ens_train(self.ens.final_ens_column_on_train)

        self.ens.set_initial_mp_ens_pred_on_val(self.ens.final_ens_pred_on_val)
        self.ens.set_initial_mp_ens_column_on_val(self.ens.final_ens_column_on_val)
        self.res_log.calculate_miss_mp_initial_ens_val(self.ens.final_ens_column_on_val)
        # /Prevents forking in ensemble method

        # miss_idx = [x for x, y in enumerate(self._u) if y > self._dual_miss_threshold]
        self._pp.update(self.x_train, self.y_train)
        ref_model, ref_preds, ref_yh, sum_uyh = self.solve_pp()

        iter = 0
        extra_iter_counter = 0
        while (sum_uyh > (self.beta + 1e-3) and iter < self._max_iter) or extra_iter_counter < self._extra_iter:
        # while iter <= self.prev_iter_n:
            if sum_uyh < (self.beta + 1e-3):
                extra_iter_counter += 1
                print(f'extra_iter_counter = {extra_iter_counter}')

            self.ens.add_refinement_H(ref_model)
            self.ens.add_refinement_pred_on_train(ref_preds)
            self.ens.add_refinement_column_on_train(ref_yh)

            self.iterate_dual()
            if logger:
                self.log_progress()

            # miss_idx = [x for x, y in enumerate(self._u) if y > self._dual_miss_threshold]
            self._pp.update(self.x_train, self.y_train)
            ref_model, ref_preds, ref_yh, sum_uyh = self.solve_pp()
            print(f'Sum of yh is {sum_uyh}')
            print(f'BETA IS: {self.beta}')
            iter += 1

        if logger and graph:
            x = self.exp_config['n_start'] - 1
            single_data = ((x, self.res_log.miss_mp_initial_ens_train, 'mp ens. train'),
                           (x, self.res_log.miss_mp_initial_ens_val, 'mp ens. val'))
            grapher = Grapher()
            grapher.make_curve(title=f'Miss-classification Scores vs. Number of BLs --' +
                                     f'({self.exp_config["bl_type"]}-{self.exp_config["bl_type_ref"]})'
                               , label='L_m', x_label='Number of BLs',
                               y_label='Miss-classification Score of the Ensemble', color='black',
                               y=self.res_log.miss_score_train_history,
                               additional_ys=[self.res_log.miss_score_val_history],
                               additional_label=['L_m*'], marker='o', single_point_data=single_data,
                               vline=self.exp_config['n_start'] - 1)

            grapher.make_curve(title='Master Problem Value History', label='L_h',
                               x_label='Refinement Iteration',
                               y_label='Master Problem Value', color='blue', y=self.res_log.mp_history, marker='o')

            grapher.make_curve(title='Average Diversity of Data Points History', label='Avg. Div.',
                               x_label='Iteration', y_label='Diversity Measure', color='green',
                               y=self.res_log.avg_diversity_history, marker='o',
                               vline=self.exp_config['n_start'] - 1)

            # for idx, diversity in enumerate(self.res_log.diversity_history):
            #     grapher.make_curve(title=f'Diversity of Data Points for Refinement Iteration: {idx}',
            #                        label=None, x_label='Datapoint', y_label='Diversity Measure',
            #                        color='green', y=diversity, marker=None, vline=self.exp_config['n_start']-1)

            grapher.show_plots()

            return 0


class MasterProblemLPBoost(MasterProblem):
    """ All training data to sub problem and sub problem is pricing problem (linear)"""
    def __init__(self, d, ens_obj: Ensemble, pp, x_train, y_train, y_train_comp, y_val, y_val_comp, exp_config,
                 extra_iter=0, result_logger=None):

        super().__init__(d, ens_obj, pp, x_train, y_train, y_train_comp, y_val, y_val_comp,
                         exp_config, extra_iter=extra_iter, result_logger=result_logger)

    def solve_pp(self):
        ref_model = self._pp.solve_pp_data()
        ref_preds = (ref_model.get_prediction(self.x_train, batch_size=None,
                                              binarize=self._binarize_preds))
        ref_yh = ref_preds * self.y_train_comp
        uyh = ref_yh * self._u
        sum_uyh = sum(uyh)

        return ref_model, ref_preds, ref_yh, sum_uyh

    def solve_in_dual(self, logger=True, graph=False, var_gen_end=None):
        # PP is initialized with all data points, explicit dual weight formulation not needed, just vector length
        self._u = np.zeros((self._n_samples, 1))

        self.iterate_dual()

        # Prevents forkings in ensemble method. In first iteration, self.ens.final_ens... equals the first mp ens
        # alphas = self.solve_in_primal()
        alphas = self._m_d.getAttr("Pi")[:-1]
        alphas = [-i for i in alphas]
        self.ens.weights = alphas
        _, _, _, _ = self.ens.get_scores()

        # Calculation of miss score for MP ini alt is not necessary, since self.ens.get_scores() already return
        # miss score
        self.ens.set_initial_mp_ens_pred_on_train(self.ens.final_ens_pred_on_train)
        self.ens.set_initial_mp_ens_column_on_train(self.ens.final_ens_column_on_train)
        self.res_log.calculate_miss_mp_initial_ens_train(self.ens.final_ens_column_on_train)

        self.ens.set_initial_mp_ens_pred_on_val(self.ens.final_ens_pred_on_val)
        self.ens.set_initial_mp_ens_column_on_val(self.ens.final_ens_column_on_val)
        self.res_log.calculate_miss_mp_initial_ens_val(self.ens.final_ens_column_on_val)
        # /Prevents forking in ensemble method

        self._pp.update_u(self._u)
        ref_model, ref_preds, ref_yh, sum_uyh = self.solve_pp()

        iter = 0
        extra_iter_counter = 0
        while (sum_uyh > (self.beta + 1e-3) and iter < self._max_iter) or extra_iter_counter < self._extra_iter:
            if sum_uyh < (self.beta + 1e-3):
                extra_iter_counter += 1
                print(f'extra_iter_counter = {extra_iter_counter}')

            self.ens.add_refinement_H(ref_model)
            self.ens.add_refinement_pred_on_train(ref_preds)
            self.ens.add_refinement_column_on_train(ref_yh)

            self.iterate_dual()
            if logger:
                self.log_progress()

            self._pp.update_u(self._u)
            ref_model, ref_preds, ref_yh, sum_uyh = self.solve_pp()
            print(f'Sum of yh is {sum_uyh}')
            print(f'BETA IS: {self.beta}')
            iter += 1

        if logger and graph:
            if var_gen_end is None:
                vline = self.exp_config['n_start'] - 1
                x = self.exp_config['n_start'] - 1
                single_point = ((x, self.res_log.miss_mp_initial_ens_train, 'mp ens. train'),
                                (x, self.res_log.miss_mp_initial_ens_val, 'mp ens. val'))
            else:
                vline = var_gen_end
                single_point = None

            self.graph_progress(single_point=single_point, vline=vline)

        return 0


class MasterProblemLPBoostActive(MasterProblem):
    """ Pricing problem with active data only"""
    def __init__(self, d, ens_obj: Ensemble, pp, x_train, y_train, y_train_comp, exp_config, extra_iter=0,
                 result_logger=None):

        super().__init__(d, ens_obj, pp, x_train, y_train, y_train_comp, exp_config, extra_iter=extra_iter,
                         result_logger=result_logger)

    def solve_in_dual(self, logger=True, graph=False, var_gen_end=None):
        # PP is initialized with all data points, explicit dual weight formulation not needed, just vector length
        self._u = np.zeros((self._n_samples, 1))

        self.iterate_dual()

        # Prevents forkings in ensemble method. In first iteration, self.ens.final_ens... equals the first mp ens
        # alphas = self.solve_in_primal()
        alphas = self._m_d.getAttr("Pi")[:-1]
        alphas = [-i for i in alphas]
        self.ens.weights = alphas
        _, _, _, _ = self.ens.get_scores()

        # Calculation of miss score for MP ini alt is not necessary, since self.ens.get_scores() already return
        # miss score
        self.ens.set_initial_mp_ens_pred_on_train(self.ens.final_ens_pred_on_train)
        self.ens.set_initial_mp_ens_column_on_train(self.ens.final_ens_column_on_train)
        self.res_log.calculate_miss_mp_initial_ens_train(self.ens.final_ens_column_on_train)

        self.ens.set_initial_mp_ens_pred_on_val(self.ens.final_ens_pred_on_val)
        self.ens.set_initial_mp_ens_column_on_val(self.ens.final_ens_column_on_val)
        self.res_log.calculate_miss_mp_initial_ens_val(self.ens.final_ens_column_on_val)
        # /Prevents forking in ensemble method

        miss_idx = [x for x, y in enumerate(self._u) if y > self._dual_miss_threshold]
        self._pp.update(self.x_train[miss_idx], self.y_train[miss_idx])
        self._pp.update_u(self._u[miss_idx])
        ref_model, ref_preds, ref_yh, sum_uyh = self.solve_pp()

        iter = 0
        extra_iter_counter = 0
        while (sum_uyh > (self.beta + 1e-3) and iter < self._max_iter) or extra_iter_counter < self._extra_iter:
            if sum_uyh < (self.beta + 1e-3):
                extra_iter_counter += 1
                print(f'extra_iter_counter = {extra_iter_counter}')

            self.ens.add_refinement_H(ref_model)
            self.ens.add_refinement_pred_on_train(ref_preds)
            self.ens.add_refinement_column_on_train(ref_yh)

            self.iterate_dual()
            if logger:
                self.log_progress()

            miss_idx = [x for x, y in enumerate(self._u) if y > self._dual_miss_threshold]
            self._pp.update(self.x_train[miss_idx], self.y_train[miss_idx])
            self._pp.update_u(self._u[miss_idx])
            ref_model, ref_preds, ref_yh, sum_uyh = self.solve_pp()
            print(f'Sum of yh is {sum_uyh}')
            print(f'BETA IS: {self.beta}')
            iter += 1

        if logger and graph:
            if var_gen_end is None:
                vline = self.exp_config['n_start'] - 1
                x = self.exp_config['n_start'] - 1
                single_point = ((x, self.res_log.miss_mp_initial_ens_train, 'mp ens. train'),
                                (x, self.res_log.miss_mp_initial_ens_val, 'mp ens. val'))
            else:
                vline = var_gen_end
                single_point = None

            self.graph_progress(single_point=single_point, vline=vline)

        return 0


class MasterProblemMissOnly(MasterProblem):
    """ Pricing problem with active data only"""
    def __init__(self, d, ens_obj: Ensemble, pp, x_train, y_train, y_train_comp, exp_config, extra_iter=0,
                 result_logger=None):

        super().__init__(d, ens_obj, pp, x_train, y_train, y_train_comp, exp_config, extra_iter=extra_iter,
                         result_logger=result_logger)

    def solve_in_dual(self, logger=True, graph=False, var_gen_end=None):

        _, _, _, _ = self.ens.get_scores()
        ens_column = self.ens.final_ens_column_on_train
        miss_idx, _ = np.where(ens_column <= 0)
        max_instances = int(1 / self._D)
        miss_idx = miss_idx[:max_instances]

        self._pp.update(self.x_train[miss_idx], self.y_train[miss_idx])
        ref_model, ref_preds, ref_yh, sum_uyh = self.solve_pp()

        iter = 0
        # extra_iter_counter = 0
        # while (sum_uyh > (self._beta + 1e-3) and iter < self._max_iter) or extra_iter_counter < self._extra_iter:
        #     if sum_uyh < (self._beta + 1e-3):
        #         extra_iter_counter += 1
        #         print(f'extra_iter_counter = {extra_iter_counter}')

        while iter <= self._extra_iter:
            self.ens.add_refinement_H(ref_model)
            self.ens.add_refinement_pred_on_train(ref_preds)
            self.ens.add_refinement_column_on_train(ref_yh)

            # Log preds on r and non_r
            self.log_preds_on_paritions(ref_preds, miss_idx)

            self.iterate_dual()
            if logger:
                self.log_progress()

            # Log u_i
            self.res_log.add_u_i(self._u)

            # Get and log rho
            # rho = self.solve_in_primal()
            self.res_log.add_margin(margin='Not_Logged')

            ens_column = self.ens.final_ens_column_on_train
            miss_idx, _ = np.where(ens_column <= 0)
            miss_idx = miss_idx[:max_instances]

            self._pp.update(self.x_train[miss_idx], self.y_train[miss_idx])
            ref_model, ref_preds, ref_yh, sum_uyh = self.solve_pp()

            # Calculate
            print(f'Sum of yh is {sum_uyh}')
            print(f'BETA IS: {self.beta}')
            iter += 1

        if logger and graph:
            if var_gen_end is None:
                vline = self.exp_config['n_start'] - 1
                x = self.exp_config['n_start'] - 1
                single_point = ((x, self.res_log.miss_mp_initial_ens_train, 'mp ens. train'),
                                (x, self.res_log.miss_mp_initial_ens_val, 'mp ens. val'))
            else:
                vline = var_gen_end
                single_point = None

            self.graph_progress(single_point=single_point, vline=vline)

        return 0