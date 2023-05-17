from __future__ import annotations
import numpy as np
import keras.metrics
from gurobipy import *
from typing import TYPE_CHECKING
from utils.grapher import Grapher
if TYPE_CHECKING:
    from ensemble.ensemble import Ensemble

import logging
logger = logging.getLogger('soda')


class MasterProblemMultiClass:
    def __init__(self, d, ens_obj: Ensemble, pp, x_train, y_train, y_val, exp_config, extra_iter=0,
                 result_logger=None):
        """Implemented in primal form only"""
        logger.info('Construct LPBoost master problem instance')
        self._D = d
        self._extra_iter = extra_iter
        self.x_train = x_train
        self.y_train = y_train
        self.y_val = y_val
        self._dual_miss_threshold = 0.0
        self._j_samples = len(x_train)
        self._m_classes = len(np.unique(y_train))
        self.ens = ens_obj
        self._pp = pp

        self._m_p = None
        self._m_d = None
        self._alpha = None
        self._mp_val = 0.0

        self._first = True

        self._u = None
        self._max_iter = 40     # 40

        # Logger #
        self.res_log = result_logger

        self._debug_mode = True

        self._GRB_beta = None
        self._GRB_u = None

        self.exp_config = exp_config

        self._new_bl_config = None

        if exp_config['comb_me'] == 'vo':
            self._binarize_preds = True
        else:
            self._binarize_preds = False

        self.iteration_counter = 0

        self._label_counter = None
        self.acc = keras.metrics.Accuracy()

    def log_progress(self, in_boosting=True):
        # Alpha stored in list
        # Flip graph
        self.res_log.add_mp_val_i(self._beta*(-1))
        slicing_idx = self._j_samples + 1
        alphas = self._m_d.getAttr("Pi")[:-slicing_idx]
        alphas = [-i for i in alphas]
        print(f'Alpha vector: {alphas}')
        self.res_log.add_alphas_i(alphas)
        self.ens.weights = alphas
        miss_score_train, miss_score_val, metrics_train, metrics_val = \
            self.ens.get_scores()

        # self.res_log.add_miss_score_train_i(miss_score_train)
        ens_acc_train = 1 - (miss_score_train / len(self.y_train))
        self.res_log.add_miss_score_train_i(ens_acc_train)

        # self.res_log.add_miss_score_val_i(miss_score_val)
        ens_acc_val = 1 - (miss_score_val / len(self.y_val))
        self.res_log.add_miss_score_val_i(ens_acc_val)

        if in_boosting:
            self.res_log.add_bl_accuracy(self.ens.refinement_H[-1].accuracy)
            self.res_log.add_multiple_y_indicator(self._label_counter)

        # add metrics to logflag
        self.res_log.add_metrics_train_i(metrics_train)
        self.res_log.add_metrics_val_i(metrics_val)

        # u_jm > 0 criteria for active points
        # Shape \bf{u}: (|J|, |M|)
        actives = np.any(self._u, axis=1)
        self.res_log.add_to_n_active_u_history(sum(actives))

        # Calculate and log diversity
        # TODO: Theorize calculating diversity for multi-class tasks
        # all_columns = self.ens.get_columns()
        # fin_ens_column_train = self.ens.final_ens_column_on_train
        #
        # avg_diversity, diversity = self.res_log.calculate_diversity(all_columns=all_columns,
        #                                                             final_ensemble_column_train=fin_ens_column_train,
        #                                                             ens_weights=alphas)
        # self.res_log.add_avg_diversity_i(avg_diversity)
        # self.res_log.add_diversity_i(diversity)

        self.res_log.add_avg_diversity_i('Not_Implemented')
        self.res_log.add_diversity_i('Not_Implemented')

        # For model comparison only
        # alphas = self.solve_in_primal()

    def graph_progress(self, vline=None, single_point=None):
        grapher = Grapher()
        grapher.make_curve(title=f'Miss-classification Scores vs. Number of BLs (Multi)--' +
                                 f'({self.exp_config["bl_type"]}-{self.exp_config["bl_type_ref"]})',
                           label='L_m', x_label='Number of BLs',
                           y_label='Miss-classification Score of the Ensemble', color='black',
                           y=self.res_log.miss_score_train_history,
                           additional_ys=[self.res_log.miss_score_val_history], additional_label=['L_m*'],
                           marker='o', single_point_data=single_point, vline=vline)

        grapher.make_curve(title='Master Problem Value History (Multi)', label='L_h',
                           x_label='Refinement Iteration', y_label='Master Problem Value', color='blue',
                           y=self.res_log.mp_history, marker='o')

        # TODO: Activate graphing for diversity when multi-class div. is implemented
        # grapher.make_curve(title='Average Diversity of Data Points History', label='Avg. Div.',
        #                    x_label='Iteration', y_label='Diversity Measure', color='green',
        #                    y=self.res_log.avg_diversity_history, marker='o', vline=vline)

        grapher.show_plots()

    def log_columns_on_paritions(self, ref_preds, miss_idx, non_miss_idx):
        # u_jm > 0 criteria for active points

        # Calculate column
        len_t_t, len_m_t = ref_preds.shape
        pred_for_true_y_train = ref_preds[range(len_t_t), self.y_train]
        # Generate sol_col
        col_with_hy = np.zeros((len_t_t, len_m_t))
        col_with_hy[range(len_t_t), self.y_train] = pred_for_true_y_train
        # Reduce sol_col to 1D
        col_with_hy_1d = col_with_hy[np.nonzero(col_with_hy)]
        col_with_hy_1d = np.expand_dims(col_with_hy_1d, axis=1)
        # Calculate difference h_y(x) - h_m(x)
        diff_col = -ref_preds + col_with_hy_1d
        # Calculate Final
        column = col_with_hy + diff_col

        # Log Column on Non-R
        ref_non_r_column = column[non_miss_idx]
        # self.res_log.add_ref_bl_column_on_non_r(ref_non_r_column)
        self.res_log.add_ref_bl_column_on_non_r(np.asarray([-1]))

        # Log preds only on active data
        ref_r_column = column[miss_idx]
        # self.res_log.add_ref_bl_column_on_r(ref_r_column)
        self.res_log.add_ref_bl_column_on_r(np.asarray([-1]))

    # Find dual mp value, data weights u, (alpha can be extracted)
    def iterate_dual(self):
        self.iteration_counter += 1
        print(f'------------ MP Iteration {self.iteration_counter} -----------')

        self._m_d = Model('LPBoost')
        # Define Variables
        # self._GRB_beta = self._m_d.addVar(vtype=GRB.CONTINUOUS, name='Beta')
        # self._GRB_beta = self._m_d.addVar(vtype=GRB.CONTINUOUS, lb=-1, ub=1, name='Beta', column=None, obj=0)
        self._GRB_beta = self._m_d.addVar(vtype=GRB.CONTINUOUS, name='Beta', lb=-1, column=None, obj=0)

        # Define u
        self._GRB_u = self._m_d.addVars(self._j_samples, self._m_classes,
                                        vtype=GRB.CONTINUOUS, name='u', lb=0)

        self._m_d.update()

        # Define Objective
        self._m_d.setObjective(self._GRB_beta, GRB.MINIMIZE)

        # TODO: Building model with Gurobi 9 is very slow: Matrix operations with native Python is not allowed
        # Beta Constraint
        for con_counter, column_row in enumerate(self.ens.get_columns(binarize=False)):
            name = f'constraint_{con_counter}'
            u_column_i = quicksum(column_row[j, m] * self._GRB_u[j, m] for j in range(self._j_samples)
                                  for m in range(self._m_classes))
            self._m_d.addConstr(u_column_i <= self._GRB_beta, name=name)

        # Lambda Constraint
        u_m = []
        for j in range(self._j_samples):
            subset_i = self._GRB_u.select(j, '*')
            subset_i.__delitem__(self.y_train[j])
            self._m_d.addConstr(quicksum(subset_i) <= self._D, name='Lambda_Constr')
            u_m.extend(subset_i)

        # Convex Combination Constraint
        self._m_d.addConstr(quicksum(u_m) == 1, name='Conv_Comb_constraint')

        # Optimize
        self._m_d.setParam("LogToConsole", 1)  # switch off solver logging in console

        self._m_d.optimize()

        print(f'MP Dual Objective: {self._m_d.objVal}')

        # # Since this model is the dual, it must be saved as .dlp to get the primal; for debugging only
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
        self._beta = vars[0].x

        self._u = []
        for var in vars[1:]:
            self._u.append(var.x)
        self._u = np.asarray(self._u)
        self._u = np.reshape(self._u, (self._j_samples, self._m_classes))

        # Linear constraint attribute, alpha for NEXT Iteration
        # print(f'Dual values in dual formulation: {self._m_d.getAttr("Pi")}')
        self._first = False

        return 0

    def solve_pp(self):
        ref_model = self._pp.solve_pp_data()
        pred = (ref_model.get_prediction(self.x_train, batch_size=None,
                                         binarize=self._binarize_preds))

        # Generate initial columns
        len_t_t, len_m_t = pred.shape
        pred_for_true_y_train = pred[range(len_t_t), self.y_train]
        # Generate sol_col
        col_with_hy = np.zeros((len_t_t, len_m_t))
        col_with_hy[range(len_t_t), self.y_train] = pred_for_true_y_train
        # Reduce sol_col to 1D
        col_with_hy_1d = col_with_hy[np.nonzero(col_with_hy)]
        col_with_hy_1d = np.expand_dims(col_with_hy_1d, axis=1)
        # Calculate difference h_y(x) - h_m(x)
        diff_col = -pred + col_with_hy_1d
        # Calculate Final
        column = col_with_hy + diff_col

        # Add vector u as coefficients
        # u_column = column_train * self._u
        u_m_column = diff_col * self._u

        sum_u_m_column = np.sum(u_m_column)

        return ref_model, pred, column, sum_u_m_column

    def solve_in_dual_for_gen(self):
        """Returns active date; intended for dynamically terminated generation phase only"""
        self.iterate_dual()

        return self._u

    def solve_in_dual(self, logger=True):

        # Identify missing idx with
        self.iterate_dual()
        miss_idx = self._pp.get_active_data_idx(u=self._u, threshold=self._dual_miss_threshold, mode='multi')

        miss_y = self.y_train[miss_idx]
        unique, counts = np.unique(miss_y, return_counts=True)
        self._label_counter = str(tuple(zip(unique, counts)))

        self._pp.update(self.x_train[miss_idx], miss_y)
        ref_model, ref_pred, ref_column, sum_u_m_column = self.solve_pp()

        iter = 0

        # while (sum_u_m_column > (self._beta + 1e-3) and iter < self._max_iter) or extra_iter_counter < self._extra_iter:
        #     if sum_u_m_column < (self._beta + 1e-3):
        #         extra_iter_counter += 1
        #         print(f'extra_iter_counter = {extra_iter_counter}')

        while iter < self._extra_iter:
            iter += 1

            self.ens.add_refinement_H(ref_model)
            self.ens.add_refinement_pred_on_train(ref_pred)
            self.ens.add_refinement_column_on_train(ref_column)

            # Log preds on r and non-r
            r_truth = np.any(self._u, axis=1)
            non_miss_idx, = np.where(r_truth==False)
            self.log_columns_on_paritions(ref_pred, miss_idx, non_miss_idx)

            self.iterate_dual()

            if logger:
                self.log_progress()
                # Log u_i
                # self.res_log.add_u_i(self._u)
                self.res_log.add_u_i(np.asarray([-1]))
                # Get and log rho
                # rho = self.solve_in_primal()
                self.res_log.add_margin(margin='Not_Logged')

                self.acc.reset_states()
                self.acc.update_state(self.y_train, np.argmax(ref_pred, axis=1))
                self.res_log.add_bl_accuracy_on_original_train(self.acc.result().numpy())

                self.acc.reset_states()
                self.acc.update_state(self.y_val, np.argmax(self.ens.refinement_preds_on_val[-1], axis=1))
                self.res_log.add_bl_accuracy_on_original_val(self.acc.result().numpy())

            miss_idx = self._pp.get_active_data_idx(self._u, self._dual_miss_threshold, mode='multi')

            miss_y = self.y_train[miss_idx]
            unique, counts = np.unique(miss_y, return_counts=True)
            self._label_counter = str(tuple(zip(unique, counts)))

            # # Remove 50% of most severe miss-classified
            # reduced_miss = self.reduce_r(miss_idx=miss_idx)
            # self._pp.update(self.x_train[reduced_miss], self.y_train[reduced_miss])

            self._pp.update(self.x_train[miss_idx], miss_y)
            if iter == self._extra_iter:
                ref_model, ref_pred, ref_column, sum_u_m_column = self.solve_pp()
            else:
                pass
            # Calculate
            print(f'Sum of Columns is {sum_u_m_column}')
            print(f'BETA IS: {self._beta}')

        return 0

    def solve_in_primal(self, logger=True, graph=False, var_gen_end=None):
        """ Description:
                - Formely solve_in_dual()
                -
        """
        self.iteration_counter += 1
        print(f'########################## Iteration {self.iteration_counter} ##################################')

        self._m_p = Model('LPBoost_Primal')

        # Add variables
        # rho = self._m_p.addVar(vtype=GRB.CONTINUOUS, name='rho', lb=-1, ub=1)
        rho = self._m_p.addVar(vtype=GRB.CONTINUOUS, name='rho')
        slacks = []
        alphas = []
        columns = self.ens.get_columns()

        for i in range(self._j_samples):
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
        for k in range(self._j_samples):
            self._m_p.addConstr(quicksum(restrictions_lhs[0][k]) + slacks[k] >= rho)
            # m_p.addConstr(restrictions_lhs[k] + slacks[k] >= rho)
        self._m_p.addConstr(quicksum(alphas) == 1)

        self._m_p.setParam("LogToConsole", 0)  # switch off solver logging in console

        self._m_p.optimize()

        # print('mp primal obj: {0}'.format(m_p.objVal))

        vars = self._m_p.getVars()
        r_value = []
        show_from = self._j_samples - 1
        show_idx = 0
        for var in vars:
            r_value.append(var.x)
            if show_idx >= show_from:
                print(var)
            show_idx += 1
        rho = vars[0].x

        # return r_value[-len(self.ens.get_columns()):]

        return rho

    def refine(self, logger=True):
        self.solve_in_dual(logger=logger)
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
        #         os.remove((os.path.join(directory_name, item))

