import os
from utils.grapher import Grapher
from utils.xml_processor import XMLProcessor
from utils.result_processor import *
from itertools import chain


curr_dir = os.getcwd()
main_dir = os.path.dirname(curr_dir)
super_dir = os.path.dirname(main_dir)
writer = XMLProcessor(main_dir + '/results')

path_string = r"C:\Users\vanya\OneDrive\Desktop\decolearn\decolearn\results\cnn_cnn__cifar10__D3A_cnn_cnn_binary_1_bag_0.5__2023-05-12_17_23_11_401215.xml"
#exp_main, exp_ref = writer.load_all(path_to_load, 2)
exp_ref = writer.load_all(path_string)
exp0 = exp_ref

bl_acc_hist_train = list()
bl_acc_hist_val = list()
ens_acc_hist_train = list()
ens_acc_hist_val = list()


for bagging_iter in exp0['data']['bagging_phase']:
    bl_acc_hist_train.append(exp0['data']['bagging_phase'][bagging_iter]['bl_acc_or_train'])
    bl_acc_hist_val.append(exp0['data']['bagging_phase'][bagging_iter]['bl_acc_or_val'])
    ens_acc_hist_train.append(exp0['data']['bagging_phase'][bagging_iter]['miss_train'])
    ens_acc_hist_val.append(exp0['data']['bagging_phase'][bagging_iter]['miss_val'])


for boosting_iter in exp0['data']['boosting_phase']:
    bl_acc_hist_train.append(exp0['data']['boosting_phase'][boosting_iter]['bl_acc_or_train'])
    bl_acc_hist_val.append(exp0['data']['boosting_phase'][boosting_iter]['bl_acc_or_val'])
    ens_acc_hist_train.append(exp0['data']['boosting_phase'][boosting_iter]['miss_train'])
    ens_acc_hist_val.append(exp0['data']['boosting_phase'][boosting_iter]['miss_val'])

print(bl_acc_hist_train)
print(bl_acc_hist_val)
print(ens_acc_hist_train)
print(ens_acc_hist_val)

max_bl_acc_train = max(bl_acc_hist_train)
max_bl_acc_val = max(bl_acc_hist_val)
max_ens_acc_train = max(ens_acc_hist_train)
max_ens_acc_val = max(ens_acc_hist_val)

print(max_bl_acc_train)
print(max_bl_acc_val)
print(max_ens_acc_train)
print(max_ens_acc_val)