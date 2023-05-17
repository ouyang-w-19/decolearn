from __future__ import annotations
import numpy as np
from gurobipy import *
from typing import TYPE_CHECKING
from utils.grapher import Grapher
if TYPE_CHECKING:
    from ensemble.ensemble import Ensemble

import logging
logger = logging.getLogger('soda')


class MasterProblem:
    """
    - Class that implements binary LPBoost with soft margin
    """
    def __init__(self, d, ens_obj: Ensemble, pp, x_train, x_val, y_train, y_train_comp, y_val, y_val_comp,
                 exp_config, extra_iter=0,
                 result_logger=None):
        logger.info('Construct LPBoost master problem instance')
        self._D = d
        self._extra_iter = extra_iter
        self.x_train = x_train
        self.x_val = x_val
        self.y_train = y_train
        self.y_val = y_val
        self.y_val_comp = y_val_comp
        self.y_train_comp = np.expand_dims(y_train_comp, axis=1)
        self._dual_miss_threshold = 0.0
        self._n_samples = len(x_train)
        self.ens = ens_obj
        self._pp = pp

        self._m_p = None
        self._m_d = None
        self.beta = 0.0

        self._first = True

        self._u = None
        self._ens_res_i = None
        self._max_iter = 40     # 50

        # Logger #
        self.res_log = result_logger
        self._GRB_beta = None
        self._GRB_u = None
        self.exp_config = exp_config
        self.multiple_y_indicator = None
        self._new_bl_config = None

        if exp_config['comb_me'] == 'vo':
            self._binarize_preds = True
        else:
            self._binarize_preds = False

        self.iteration_counter = 0

    def log_progress(self, in_boosting=True):
        # Alpha stored in list
        # Flip graph
        self.res_log.add_mp_val_i(self.beta * (-1))
        alphas = self._m_d.getAttr("Pi")[:-1]
        alphas = [-i for i in alphas]
        self.res_log.add_alphas_i(alphas)
        self.ens.weights = alphas
        miss_score_train, miss_score_val, metrics_train, metrics_val = \
            self.ens.get_scores()

        # self.res_log.add_miss_score_train_i(miss_score_train)
        # self.res_log.add_miss_score_val_i(miss_score_val)

        # self.res_log.add_miss_score_train_i(miss_score_train)
        ens_acc_train = 1 - (miss_score_train / len(self.y_train))
        self.res_log.add_miss_score_train_i(ens_acc_train)

        # self.res_log.add_miss_score_val_i(miss_score_val)
        ens_acc_val = 1 - (miss_score_val / len(self.y_val_comp))
        self.res_log.add_miss_score_val_i(ens_acc_val)

        if in_boosting:
            self.res_log.add_bl_accuracy(self.ens.refinement_H[-1].accuracy)
            # Add indicator for multiple y
            self.res_log.add_multiple_y_indicator(self.multiple_y_indicator)

        # add metrics to logflag
        self.res_log.add_metrics_train_i(metrics_train)
        self.res_log.add_metrics_val_i(metrics_val)

        self.res_log.add_to_n_active_u_history(sum(np.where(self._u > 0, 1, 0)))

        # Calculate and log diversity
        all_yhs = self.ens.get_columns()
        final_ensemble_yh_train = self.ens.final_ens_column_on_train
        # weights = self.ens.weights
        avg_diversity, diversity = self.res_log.calculate_binary_diversity(all_columns=all_yhs,
                                                                           final_ensemble_column_train=final_ensemble_yh_train,
                                                                           ens_weights=alphas)
        self.res_log.add_avg_diversity_i(avg_diversity)
        self.res_log.add_diversity_i(diversity)

        # For model comparison only
        # alphas = self.solve_in_primal()

    def graph_progress(self, vline=None, single_point=None):
        grapher = Grapher()
        grapher.make_curve(title=f'Miss-classification Scores vs. Number of BLs --' +
                                 f'({self.exp_config["bl_type"]}-{self.exp_config["bl_type_ref"]})',
                           label='L_m', x_label='Number of BLs',
                           y_label='Miss-classification Score of the Ensemble', color='black',
                           y=self.res_log.miss_score_train_history,
                           additional_ys=[self.res_log.miss_score_val_history], additional_label=['L_m*'],
                           marker='o', single_point_data=single_point, vline=vline)

        grapher.make_curve(title='Master Problem Value History', label='L_h',
                           x_label='Refinement Iteration', y_label='Master Problem Value', color='blue',
                           y=self.res_log.mp_history, marker='o')

        grapher.make_curve(title='Average Diversity of Data Points History', label='Avg. Div.',
                           x_label='Iteration', y_label='Diversity Measure', color='green',
                           y=self.res_log.avg_diversity_history, marker='o', vline=vline)

        grapher.show_plots()

    def log_preds_on_paritions(self, ref_preds, miss_idx):
        # Log preds only on non-active data
        non_r = [x for x, y in enumerate(self._u) if y == 0]
        ref_preds_non_r_column = ref_preds[non_r] * self.y_train_comp[non_r]
        self.res_log.add_ref_bl_column_on_non_r(ref_preds_non_r_column)

        # Log preds only on active data
        ref_preds_r_column = ref_preds[miss_idx] * self.y_train_comp[miss_idx]
        self.res_log.add_ref_bl_column_on_r(ref_preds_r_column)

    # Find dual mp value, data weights u, (alpha can be extracted)
    def iterate_dual(self):
        self.iteration_counter += 1
        print(f'------------ MP Iteration {self.iteration_counter} -----------')

        self._m_d = Model('LPBoost')
        # Define Variables
        # self._GRB_beta = self._m_d.addVar(vtype=GRB.CONTINUOUS, name='Beta')
        self._GRB_beta = self._m_d.addVar(vtype=GRB.CONTINUOUS, lb=-1, ub=1, name='Beta')
        self._GRB_u = []
        for i in range(self._n_samples):
            self._GRB_u.append(self._m_d.addVar(lb=0.0, ub=self._D, vtype=GRB.CONTINUOUS))
        self._m_d.update()

        # Define Objective
        self._m_d.setObjective(self._GRB_beta, GRB.MINIMIZE)

        # Create constraints from all generated scores: Newer version: Initial ensemble not aggregated + ref. yhs
        for column_row in self.ens.get_columns(binarize=False):
            idx_c = 0
            temp = []
            for sample in column_row:
                temp.append(sample * self._GRB_u[idx_c])
                # print(temp[idx_c])
                idx_c += 1
            temp = np.asarray(temp)
            self._m_d.addConstr(quicksum(temp.squeeze()) <= self._GRB_beta)

        self._m_d.addConstr(quicksum(self._GRB_u) == 1)

        # Optimize
        self._m_d.setParam("LogToConsole", 0)  # switch off solver logging in console

        self._m_d.optimize()

        print('mp dual obj: {0}'.format(self._m_d.objVal))

        # # Since this model is the dual, it must be saved as .dlp to get the primal
        # if self._debug_mode:
        #     folder_name = 'results/'
        #     file_name_primal = f'Models_Primal_{self.exp_config["bl_type"]}-{self.exp_config["bl_type_ref"]}_' + \
        #                 f'iteration{self._iteration_counter // 2 + 1}.dlp'
        #     file_name_dual = f'Models_Dual_{self.exp_config["bl_type"]}-{self.exp_config["bl_type_ref"]}_' + \
        #                 f'iteration{self._iteration_counter // 2 + 1}.lp'
        #     file_name_primal_complete = folder_name + file_name_primal
        #     file_name_dual_complete = folder_name + file_name_dual
        #     self._m_d.write(file_name_primal_complete)
        #     self._m_d.write(file_name_dual_complete)

        vars = self._m_d.getVars()
        # for var in vars:
        #     print(var.varname, var.x)
        self.beta = vars[0].x
        for i in range(self._n_samples):
            self._u[i] = vars[i + 1].x     # (10000, 1)

        # Linear constraint attribute, alpha for NEXT Iteration
        print(f'Dual values in dual formulation: {self._m_d.getAttr("Pi")}')
        self._first = False

        return 0

    def solve_pp(self):
        ref_model = self._pp.solve_pp_data()
        ref_preds = (ref_model.get_prediction(self.x_train, batch_size=None,
                                              binarize=self._binarize_preds))
        ref_preds_mapped = -1 + 2 * ref_preds
        ref_yh = ref_preds_mapped * self.y_train_comp
        uyh = ref_yh * self._u
        sum_uyh = sum(uyh)

        return ref_model, ref_preds_mapped, ref_yh, sum_uyh

    def solve_in_dual_for_gen(self):
        """Returns active date and calculates the current final ensemble values"""
        self._u = np.zeros((self._n_samples, 1))
        self.iterate_dual()

        return self._u

    def reduce_r(self, miss_idx):
        # Remove 50% of most severe miss-classified
        # d.items() : Gets the key-value-pair
        d_ens_out = {key: value for key, value in enumerate(self.ens.final_ens_column_on_train)}
        d_r = dict()
        for k, v in d_ens_out.items():
            if k in miss_idx:
                d_r[k] = v
        d_ordered_r = dict(sorted(d_r.items(), key=lambda item: item[1]))
        l_ordered_r = list(d_ordered_r)
        half_length = int(len(l_ordered_r) * 0.5)
        reduced_miss = l_ordered_r[-half_length:]

        return reduced_miss

    def solve_in_dual(self, logger=True, graph=False, var_gen_end=None):

        miss_idx = self._pp.get_active_data_idx(self._u, self._dual_miss_threshold, mode='binary')

        # # Remove 50% of most severe miss-classified
        # reduced_miss = self.reduce_r(miss_idx=miss_idx)
        # self._pp.update(self.x_train[reduced_miss], self.y_train[reduced_miss])

        miss_y_comp = self.y_train_comp[miss_idx]
        # if len(np.unique(miss_y)) == 1:
        #     self.multiple_y_indicator = 'False'
        # else:
        #     self.multiple_y_indicator = 'True'
        self.multiple_y_indicator = str(sum(miss_y_comp) / len(miss_y_comp))

        self._pp.update(self.x_train[miss_idx], self.y_train[miss_idx])
        ref_model, ref_preds, ref_yh, sum_uyh = self.solve_pp()

        iter = 0
        # extra_iter_counter = 0
        # while (sum_uyh > (self._beta + 1e-3) and iter < self._max_iter) or extra_iter_counter < self._extra_iter:
        #     if sum_uyh < (self._beta + 1e-3):
        #         extra_iter_counter += 1
        #         print(f'extra_iter_counter = {extra_iter_counter}')

        while iter < self._extra_iter:
            self.ens.add_refinement_H(ref_model)
            self.ens.add_refinement_pred_on_train(ref_preds)
            self.ens.add_refinement_column_on_train(ref_yh)

            # Log preds on r and non_r
            self.log_preds_on_paritions(ref_preds, miss_idx)

            # Log multiple y indicator
            # if len(np.unique(self.y_train)

            self.iterate_dual()
            if logger:
                self.log_progress()

                # Log Train
                miss_score = np.where(ref_yh <= 0, 1, 0)
                bl_acc_train = 1 - (sum(miss_score) / len(self.y_train_comp))
                self.res_log.add_bl_accuracy_on_original_train(bl_acc_train)

                # Log Val
                scaled_pred = -1 + 2 * ref_model.get_prediction(self.x_val)
                column_val = np.expand_dims(self.y_val_comp, axis=1) * scaled_pred
                miss = np.where(column_val <= 0, 1, 0)
                acc_val = 1 - (sum(miss) / len(self.y_val_comp))
                self.res_log.add_bl_accuracy_on_original_val(acc_val)

            # Log u_i
            self.res_log.add_u_i(self._u)

            # Get and log rho
            # rho = self.solve_in_primal()
            self.res_log.add_margin(margin='Not_Logged')

            miss_idx = self._pp.get_active_data_idx(self._u, self._dual_miss_threshold, mode='binary')
            miss_y = self.y_train_comp[miss_idx]
            # if len(np.unique(miss_y)) == 1:
            #     self.multiple_y_indicator = 'False'
            # else:
            #     self.multiple_y_indicator = 'True'
            self.multiple_y_indicator = str(sum(miss_y) / len(miss_y))

            # # Remove 50% of most severe miss-classified
            # reduced_miss = self.reduce_r(miss_idx=miss_idx)
            # self._pp.update(self.x_train[reduced_miss], self.y_train[reduced_miss])

            self._pp.update(self.x_train[miss_idx], self.y_train[miss_idx])
            ref_model, ref_preds, ref_yh, sum_uyh = self.solve_pp()

            # Calculate
            print(f'Sum of yh is {sum_uyh}')
            print(f'BETA IS: {self.beta}')
            iter += 1

        return 0

    def solve_in_primal(self):
        self.iteration_counter += 1
        print(f'########################## Iteration {self.iteration_counter} ##################################')

        self._m_p = Model('LPBoost_Primal')

        # Add variables
        # rho = self._m_p.addVar(vtype=GRB.CONTINUOUS, name='rho', lb=-1, ub=1)
        rho = self._m_p.addVar(vtype=GRB.CONTINUOUS, name='rho')
        slacks = []
        alphas = []
        columns = self.ens.get_columns()

        for i in range(self._n_samples):
            slacks.append(self._m_p.addVar(vtype=GRB.CONTINUOUS, name='Xi_' + str(i)))

        for i in range(len(columns)):
            alphas.append(self._m_p.addVar(vtype=GRB.CONTINUOUS, name='Alpha_' + str(i)))

        alphas = np.asarray(alphas)
        self._m_p.update()

        # Set Objective
        self._m_p.setObjective(rho - (self._D * quicksum(slacks)), GRB.MAXIMIZE)

        # Add Constraints
        yh_t = self.ens.get_columns().T
        restrictions_lhs = yh_t * alphas
        for k in range(self._n_samples):
            self._m_p.addConstr(quicksum(restrictions_lhs[0][k]) + slacks[k] >= rho)
            # m_p.addConstr(restrictions_lhs[k] + slacks[k] >= rho)
        self._m_p.addConstr(quicksum(alphas) == 1)

        self._m_p.setParam("LogToConsole", 0)  # switch off solver logging in console

        self._m_p.optimize()

        # print('mp primal obj: {0}'.format(m_p.objVal))

        vars =self._m_p.getVars()
        r_value = []
        show_from = self._n_samples - 1
        show_idx = 0
        for var in vars:
            r_value.append(var.x)
            if show_idx >= show_from:
                print(var)
            show_idx += 1
        rho = vars[0].x

        # return r_value[-len(self.ens.get_columns()):]
        return rho

    def refine(self, logger=False, graph=False, var_gen_end=None):
        self.solve_in_dual(logger=logger, graph=graph, var_gen_end=var_gen_end)
        if not logger:
            alphas = self.solve_in_primal()
            self.ens.weights = alphas
            return alphas
        if self.ens.weights is None:
            self.ens.weights = [1]

        # # remove files that are left
        # directory_name = os.getcwd()
        # files = os.listdir(directory_name)
        # for item in files:
        #     if item.endswith('.nl') or item.endswith('.sol') or \
        #             item.endswith('.log') or item.endswith('.lp') \
        #             or item.endswith('.dlp'):
        #         os.remove((os.path.join(directory_name, item)))
