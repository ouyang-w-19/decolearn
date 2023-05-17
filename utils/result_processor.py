import numpy as np

"""Collections of tools to further process result data"""


def get_binary_partition_hists(exp0, last_iter_key):
    # Histogram
    miss_non_r_hist = []
    miss_r_hist = []
    acc_non_r_hist = []
    acc_r_hist = []
    balance_hist = []

    len_non_r = len(exp0['data']['boosting_phase'][last_iter_key]['ref_column_non_r'])
    # len_r = len(exp0['data']['boosting_phase'][last_iter_key]['ref_column_r'])

    for idx, iter in enumerate(exp0['data']['boosting_phase']):
        non_r = np.asarray(exp0['data']['boosting_phase'][iter]['ref_column_non_r'])
        r = np.asarray(exp0['data']['boosting_phase'][iter]['ref_column_r'])
        balance_i = exp0['data']['boosting_phase'][iter]['y_balance']

        miss_occ_non_r = np.where(non_r <= 0, 1, 0)
        miss_occ_r = np.where(r <= 0, 1, 0)

        miss_non_r = sum(miss_occ_non_r)
        miss_r = sum(miss_occ_r)
        acc_non_r = 1 - (miss_non_r / len_non_r)
        acc_r = 1 - (miss_r / len(r))

        miss_non_r_hist.append(list(miss_non_r))
        miss_r_hist.append(list(miss_r))
        acc_non_r_hist.append(list(acc_non_r))
        acc_r_hist.append(list(acc_r))
        balance_hist.append(list(balance_i))

    return miss_non_r_hist, miss_r_hist, acc_non_r_hist, acc_r_hist


def get_binary_balance_hist(exp0):
    balance_hist = []

    for idx, iter in enumerate(exp0['data']['boosting_phase']):
        balance_i = exp0['data']['boosting_phase'][iter]['y_balance']
        balance_hist.append(list(balance_i))

    return balance_hist


def get_multi_partition_hists(exp0, last_iter_key):
    miss_non_r_hist = []
    miss_r_hist = []
    acc_non_r_hist = []
    acc_r_hist = []
    balance_hist = []

    len_non_r = len(exp0['data']['boosting_phase'][last_iter_key]['ref_column_non_r'])
    # len_r = len(exp0['data']['boosting_phase'][last_iter_key]['ref_column_r'])

    for idx, iter in enumerate(exp0['data']['boosting_phase']):
        column_non_r = np.asarray(exp0['data']['boosting_phase'][iter]['ref_column_non_r'])
        column_r = np.asarray(exp0['data']['boosting_phase'][iter]['ref_column_r'])
        balance_i = exp0['data']['boosting_phase'][iter]['y_balance']

        truth_matrix_non_r = column_non_r < 0
        miss_score_non_r = sum([1 if any(row) else 0 for row in truth_matrix_non_r])

        truth_matrix_r = column_r < 0
        miss_score_r = sum([1 if any(row) else 0 for row in truth_matrix_r])

        acc_non_r = 1 - (miss_score_non_r / len_non_r )
        acc_r = 1 - (miss_score_r / len(column_r))

        miss_non_r_hist.append([miss_score_non_r])
        miss_r_hist.append([miss_score_r])
        acc_non_r_hist.append([acc_non_r])
        acc_r_hist.append([acc_r])
        balance_hist.append([balance_i])

    return miss_non_r_hist, miss_r_hist, acc_non_r_hist, acc_r_hist


def get_miss_score_multi(pred: np.ndarray, y_true: np.ndarray, pred_as_vector=True) -> np.ndarray:
    """
    Calculates the miss-classification score for multi-classification models
    :param pred:Prediction of classifier
    :param y_true: True labels of datapoint
    :param pred_as_vector: Whether prediction is given as 1D or 2D array
    :return:
    """
    if pred_as_vector:
        pred = np.argmax(pred, axis=1)
    miss_score_hist = np.where(pred==y_true, 0, 1)
    miss_score = np.sum(miss_score_hist)

    return miss_score


def get_miss_idx(pred_train: np.ndarray, y_true: np.ndarray, preds_as_vector=True) -> np.ndarray:
    """
    Returns the indices where the predictions and true labels deviate
    :param pred_train: Prediction of an approximator
    :param y_true: True labels of datapoints
    :param preds_as_vector: Whether prediction is given as 1D or 2D array
    :return: Indices of miss-classified datapoints
    """
    if preds_as_vector:
        max_pred = np.argmax(pred_train, axis=1)
        miss_idx, = np.where(max_pred != y_true)
    else:
        miss_idx, = np.where(pred_train != y_true)

    return miss_idx

