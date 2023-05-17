import os
from utils.grapher import Grapher
from utils.result_processor import *
from utils.xml_processor import XMLProcessor
from itertools import chain


curr_dir = os.getcwd()
main_dir = os.path.dirname(curr_dir)
super_dir = os.path.dirname(main_dir)
writer = XMLProcessor(main_dir + '/results')

xml_name = 'cnn_cnn__cifar10__D3A_cnn_cnn_multi_1_bag_0.5__2023-04-08_17_00_16_147581'
# middle_path = '/ResidualFocusedLPBoost_vs_ResidualFocusedLPBoostActiveData/cnn_cnn_cifar10_t3_nt2/'
# middle_path = 'ResidualFocusedVanilla_vs_ResidualFocusedLPBoost/CNN_CNN_cifar10_t3_nt2/'
middle_path = ''
path_to_load = super_dir + '/results/' + middle_path + xml_name + '.xml'
#exp_main, exp_ref = writer.load_all(path_to_load, 2)

# 25s with 6 Columns
exp_ref = writer.load_all(path_to_load)
exp0 = exp_ref

### Extract and graph the data ###mlp_mlp__mnist_20000__2022-06-23_14_08_44_188680

graph_all_div_data = False

mp_val_hist = []
miss_train_hist = []
miss_val_hist = []
avg_div_hist = []
div_datapoints_hist = []
alpha_hist = []

all_hist = [miss_train_hist, miss_val_hist, mp_val_hist, avg_div_hist, div_datapoints_hist]

alternative_mp_train = exp0['data']['alt_ini_mp_ens_train']
alternative_mp_val = exp0['data']['alt_ini_mp_ens_val']

for bagging_iter in exp0['data']['bagging_phase']:
    # mp_val_hist.append(exp0['data']['bagging_phase'][bagging_iter]['mp_value'])
    miss_train_hist.append(exp0['data']['bagging_phase'][bagging_iter]['miss_train'])
    miss_val_hist.append(exp0['data']['bagging_phase'][bagging_iter]['miss_val'])
    avg_div_hist.append(exp0['data']['bagging_phase'][bagging_iter]['avg_div'])
    alpha_hist.append(exp0['data']['bagging_phase'][bagging_iter]['alpha'])


for boosting_iter in exp0['data']['boosting_phase']:
    mp_val_hist.append(exp0['data']['boosting_phase'][boosting_iter]['mp_value'])
    miss_train_hist.append(exp0['data']['boosting_phase'][boosting_iter]['miss_train'])
    miss_val_hist.append(exp0['data']['boosting_phase'][boosting_iter]['miss_val'])
    avg_div_hist.append(exp0['data']['boosting_phase'][boosting_iter]['avg_div'])
    alpha_hist.append(exp0['data']['boosting_phase'][boosting_iter]['alpha'])


grapher = Grapher()

# The initial model is also considered in the bagging phase in the XML file, in order to
# get the line onto the point where the phase changed, it has to be set back one iteration.
fin_bag_x = len(exp0['data']['bagging_phase']) - 0.9999999
# fin_bag_x = len(exp0['data']['bagging_phase'])
miss_score_history_title = 'Miss-classification history -- Fitted on Active Data'
if exp0['n_start'] != exp0['data']['bagging_phase']:
    single_points = None
else:
    single_points = ((fin_bag_x, alternative_mp_train, 'alt_train'), (fin_bag_x, alternative_mp_val, 'alt_val'))
grapher.make_curve_subplot(title=miss_score_history_title, label='train', x_label='Number of BLs', color='black',
                           y_label='Miss-Score', y=miss_train_hist, additional_ys=(miss_val_hist,),
                           additional_labels=('val',), marker='o', single_point_data=single_points, vline=fin_bag_x)

grapher.make_curve(title='Master Problem History', label='MP', x_label='Iteration', color='blue',
                   y_label='Master Problem value',
                   y=mp_val_hist, marker='o')


# Alpha Histogram
# for idx, alpha in enumerate(alpha_hist):
#     base_str = ''
#     title_i = f'BL Weights, Iteration {idx}'
#     grapher.make_bar(base_str, title_i, alpha)
# Graph every third alpha histogram
for idx, alpha in enumerate(alpha_hist):
    # if idx % 3 == 0:
    base_str = ''
    title_i = f'BL Weights, Iteration {idx}'
    grapher.make_bar(base_str, title_i, alpha)

# If the Boosting phase has not been entered, then take div from last bagging iter, else from last boosting iter
if len(exp0['data']['boosting_phase']) == 1 and \
        exp0['data']['boosting_phase'][f'iter_{int(fin_bag_x+1)}']['div_datapoints'] == ' ':
    last_iter_key = f'iter_{int(fin_bag_x)}'
else:
    last_iter_key = f'iter_{int(fin_bag_x)+1}'

#############################
### Analysis on T\R and R ###
#############################

# Histogram

if exp0['data']['boosting_phase'][last_iter_key]['bl_acc_i'] != ' ':

    miss_non_r_hist, miss_r_hist, acc_non_r_hist, acc_r_hist = get_multi_partition_hists(exp0, last_iter_key)
    # balance_hist = get_multi_balance_hist(exp0)

    grapher.make_bar('h', 'Miss-classified on T\R (Refinement Phase Only)',
                     list(chain.from_iterable(miss_non_r_hist)), color='orange', iter_start=int(fin_bag_x+1))
    grapher.make_bar('h', 'Miss-classified on R (Refinement Phase Only)',
                     list(chain.from_iterable(miss_r_hist)), color='orange', iter_start=int(fin_bag_x+1))
    grapher.make_bar('h', 'Accuracy on T\R (Refinement Phase Only)',
                     list(chain.from_iterable(acc_non_r_hist)), color='darkred', iter_start=int(fin_bag_x+1))
    grapher.make_bar('h', 'Accuracy on R (Refinement Phase Only)',
                     list(chain.from_iterable(acc_r_hist)), color='darkred', iter_start=int(fin_bag_x+1))
    # grapher.make_bar('y', 'Category Balance in R', list(chain.from_iterable(balance_hist)), color='blue',
    #                  iter_start=int(fin_bag_x+1))




grapher.show_plots()
