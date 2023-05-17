import os
from utils.grapher import Grapher
from utils.xml_processor import XMLProcessor
from utils.result_processor import *
from itertools import chain


curr_dir = os.getcwd()
main_dir = os.path.dirname(curr_dir)
super_dir = os.path.dirname(main_dir)
writer = XMLProcessor(main_dir + '/results')

path_string = r"C:\Users\vanya\OneDrive\Desktop\Temp\mlp_mlp_activation_test_Rescale.xml"
path_string = path_string.replace("\\", "/")
#exp_main, exp_ref = writer.load_all(path_to_load, 2)
exp_ref = writer.load_all(path_string)
exp0 = exp_ref

# Titles
miss_score_history_title = 'Miss-classification history -- Fitted on Active Data'
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
if exp0['n_start'] != exp0['data']['bagging_phase']:
    single_points = None
else:
    single_points = ((fin_bag_x, alternative_mp_train, 'alt_train'), (fin_bag_x, alternative_mp_val, 'alt_val'))
# grapher.make_curve(title='Miss-score History', label='train', x_label='Number of BLs', color='black',
#                    y_label='Miss-Score', y=miss_train_hist, additional_ys=[miss_val_hist], additional_label=['val'],
#                    marker='o', single_point_data=single_points, vline=fin_bag_x)
grapher.make_curve_subplot(title=miss_score_history_title, label='train', x_label='Number of BLs', color='black',
                           y_label='Miss-Score', y=miss_train_hist, additional_ys=(miss_val_hist,),
                           additional_labels=('val',), marker='o', single_point_data=single_points, vline=fin_bag_x)

grapher.make_curve(title='Master Problem History', label='MP', x_label='Iteration', color='blue',
                   y_label='Master Problem value',
                   y=mp_val_hist, marker='o')

# Average Diversity
if avg_div_hist[-1] == ' ':
    avg_div_hist = avg_div_hist[:-1]
grapher.make_curve(title='Average Diversity History', label='Avg. Div.', x_label='Iteration', y_label='Average Diversity',
                   color='green', y=avg_div_hist, marker='o', vline=fin_bag_x)

if graph_all_div_data:
    for idx, div_datapoints in enumerate(div_datapoints_hist):
        grapher.make_curve(title=f'Average Diversity History, Iter. {idx}', label='Div.', x_label='Iteration',
                           y_label='Average Diversity', color='green', y=div_datapoints, marker=None)

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


# Create historgram of diversity
# If the Boosting phase has not been entered, then take div from last bagging iter, else from last boosting iter
if len(exp0['data']['boosting_phase']) == 1 and \
        exp0['data']['boosting_phase'][f'iter_{int(fin_bag_x+1)}']['div_datapoints'] == ' ':
    last_iter_key = f'iter_{int(fin_bag_x)}'
    newest_div = exp0['data']['bagging_phase'][last_iter_key]['div_datapoints']
else:
    last_iter_key = f'iter_{int(fin_bag_x)+1}'
    newest_div = exp0['data']['boosting_phase'][last_iter_key]['div_datapoints']
title = 'Freqeuncy of Diversity Datapoints Grouped by Intervals'
grapher.make_freq_couter(newest_div, title, 10, 0.1, [-0.5, 0.5], pos='edge', x_label='Diversity Intervalls',
                         y_label='Number of Datapoints')

"""
Analysis on T\R and R
"""

# Histogram
# bl_accuracy_i
if exp0['data']['boosting_phase'][last_iter_key]['mp_value'] != ' ':

    miss_non_r_hist, miss_r_hist, acc_non_r_hist, acc_r_hist = get_binary_partition_hists(exp0, last_iter_key)
    balance_hist = get_binary_balance_hist(exp0)

    grapher.make_bar('h', 'Miss-classified on T\R (Refinement Phase Only)',
                     list(chain.from_iterable(miss_non_r_hist)), color='orange', iter_start=int(fin_bag_x+1))
    grapher.make_bar('h', 'Miss-classified on R (Refinement Phase Only)',
                     list(chain.from_iterable(miss_r_hist)), color='orange', iter_start=int(fin_bag_x+1))
    grapher.make_bar('h', 'Accuracy on T\R (Refinement Phase Only)',
                     list(chain.from_iterable(acc_non_r_hist)), color='darkred', iter_start=int(fin_bag_x+1))
    grapher.make_bar('h', 'Accuracy on R (Refinement Phase Only)',
                     list(chain.from_iterable(acc_r_hist)), color='darkred', iter_start=int(fin_bag_x+1))
    grapher.make_bar('y', 'Category Balance in R', list(chain.from_iterable(balance_hist)), color='blue',
                     iter_start=int(fin_bag_x+1))


grapher.show_plots()

