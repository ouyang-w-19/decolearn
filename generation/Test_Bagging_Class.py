from generation.generation import EnhancedBagging
from ensemble.ensemble import Ensemble
from utils.result_logger import ResultLogger

import numpy as np
import os

# Bagging.__name__ = 'Test'
# EnhancedBagging.__name__ = 'Test'


curr_dir = os.getcwd()
main_dir = os.path.dirname(curr_dir)

x_train = np.load(main_dir + '/dataset/mnist/x.npy') / 255
y_train_raw = np.load(main_dir + '/dataset/mnist/y.npy')
y_train = np.asarray([1 if x == 2 else 0 for x in y_train_raw])
y_train_comp = np.asarray([1 if x == 2 else -1 for x in y_train_raw])

x_val = np.load(main_dir + '/dataset/mnist/x_val.npy') / 255
y_val = np.load(main_dir + '/dataset/mnist/y_val.npy')
y_val_comp = np.asarray([1 if x == 2 else -1 for x in y_val])


max_data = 60000
comb_me = 'av'

n_start_enhanced = 20
res_logger = ResultLogger()
ens = Ensemble(x_train[:max_data], x_val, y_train_comp[:max_data], y_val_comp, n_start_enhanced, comb_me)
bl_config = ((28, 28), (3, 2, 1), 0.001, 100, 5, 'binary_crossentropy', 0)
bagging = EnhancedBagging(x_train[:max_data], y_train[:max_data], y_train_comp[:max_data],
                          x_val, y_val_comp, 'mlp', bl_config, n_start_enhanced, comb_me, ens,
                          result_logger=res_logger)
bagging.get_bl_preds_on_val()
bagging.update_ini_bagging_ens()
bagging.compare()
bagging.get_bagging_perf(graph=True)

# n_start_vanilla = 3
# ens_vanilla = Ensemble(x_train[:max_data], x_val, y_train_comp[:max_data], y_val_comp, n_start_vanilla, comb_me)
# bl_config_vanilla = ((28, 28), (3, 2, 1), 0.001, 100, 5, 'binary_crossentropy', 0)
# bagging_vanilla = Bagging(x_train[:max_data], y_train[:max_data], y_train_comp[:max_data],
#                           x_val, y_val_comp, 'mlp', bl_config_vanilla, n_start_vanilla, comb_me, ens_vanilla,
#                           debug_mode=True)
# bagging_vanilla.get_bl_preds_on_val()
# bagging_vanilla.update_ini_bagging_ens()
# bagging_vanilla.compare()
# bagging_vanilla.get_bagging_perf()










