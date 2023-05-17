import time
from datetime import datetime
from lpboost_configs import *
from lp_boosting_modified.lp_algorithm import *
from utils.xml_processor import XMLProcessor
from utils.result_logger import ResultLogger
from utils.data_loader import DataLoader


if __name__ == '__main__':
    ''' Standard run with D3A and no comparison to TD3A   
            selector = 2
            names = (RFV,)
            skip_main_ref = True
    '''
    experiment = cnn_cnn_exp_binary
    # experiment = cnn_cnn_exp_binary
    # experiment = mlp_mlp_exp
    # experiment = tree_tree_exp
    binary_mode = experiment['binary_mode']
    save_data_set = False
    only_two = False
    non_target = 2
    xml_names = ('BinaryRFV',)
    file_name = f'D3A_cnn_cnn_binary_1_bag_{experiment["nu"]}'
    curr_dir = os.getcwd()
    main_dir = os.path.dirname(curr_dir)
    save_folder = main_dir + '/decolearn_/results/'
    res_logger = ResultLogger()
    dataset = experiment['dataset']
    extra_iter = 1      # Old: 5
    time_measure = True

    """
    ####################
    # Helper Functions #
    ####################
    """

    def execute_login(logger_type: ResultLogger, ens: Ensemble):
        logger_type.set_initial_preds_on_train(ens.initial_preds_on_train)
        logger_type.set_initial_preds_on_val(ens.initial_preds_on_val)
        logger_type.set_refinement_preds_on_train(ens.refinement_preds_on_train)
        logger_type.set_refinement_preds_on_val(ens.refinement_preds_on_val)

    """
    #####################
    # /Helper Functions # 
    #####################
    """

    if binary_mode:
        modi = ['Vanilla',
                'RFV',
                'RFLPBoost',
                'RFLPBoostActiveData',
                'MissOnly']
        selector = 1
        modus = modi[selector]

        len_names = len(xml_names)
    else:
        # modus = 'MultiClass'
        modus = 'MultiClassRFV'

    data_loader = DataLoader(dataset_name=dataset, binary_labels=binary_mode, target=experiment['target_'],
                             non_target=non_target, soda_main_dir=curr_dir)

    x_raw, y_raw, x_raw_val, y_raw_val = data_loader.load_data(load_old_binary=True)

    if modus == 'Vanilla':
        main_algo = DecolearnAlgorithm(experiment, res_logger, x_raw, y_raw, x_raw_val,
                                       y_raw_val, logflag=True, extra_iter=extra_iter, save_t=save_data_set)
    elif modus == 'RFV':
        main_algo = DecolearnAlgorithmDataReductionFocused(experiment, res_logger, x_raw, y_raw, x_raw_val,
                                                           y_raw_val, logflag=True, extra_iter=extra_iter,
                                                           save_t=save_data_set)
    elif modus == 'RFLPBoost':
        main_algo = DecolearnAlgorithmLPBoost(experiment, res_logger, x_raw, y_raw, x_raw_val, y_raw_val, logflag=True,
                                              extra_iter=extra_iter, save_t=save_data_set)
    elif modus == 'RFLPBoostActiveData':
        main_algo = DecolearnAlgorithmLPBoostActiveData(experiment, res_logger, x_raw, y_raw, x_raw_val, y_raw_val,
                                                        logflag=True, extra_iter=extra_iter, save_t=save_data_set)
    elif modus == 'MissOnly':
        main_algo = DecolearnAlgorithmMissOnly(experiment, res_logger, x_raw, y_raw, x_raw_val, y_raw_val, logflag=True,
                                               extra_iter=extra_iter, save_t=save_data_set)
    elif modus == 'MultiClass':
        main_algo = DecolearnAlgorithmMulti(experiment, res_logger, x_raw, y_raw, x_raw_val, y_raw_val, logflag=True,
                                            extra_iter=extra_iter, save_t=save_data_set)
    elif modus == 'MultiClassRFV':
        main_algo = DecolearnAlgorithmMultiDataReductionFocused(experiment, res_logger, x_raw, y_raw, x_raw_val,
                                                                y_raw_val, logflag=True, extra_iter=extra_iter,
                                                                save_t=save_data_set)
    else:
        raise Exception

    if time_measure:
        if binary_mode:
            # Test if only alternative refinement is executed
            # Times only for refinement phase
            main_algo.generate()

            main_start = time.perf_counter()
            main_algo.refine(graph=False)
            main_end = time.perf_counter()
            main_time = main_end - main_start

            print(f'Time taken main: {main_time}')
            res_logger.set_algo_perf_time(main_time)
        else:
            # Time for generation + refinement is measured, since no variation is tested
            main_start = time.perf_counter()
            main_algo.generate()
            main_algo.refine()
            main_end = time.perf_counter()
            main_time = main_end - main_start
            main_algo.res_logger.set_algo_perf_time(main_time)
    else:
        # Time is not measured
        if binary_mode:
            main_algo.generate()
            main_algo.refine(graph=True)
        else:
            main_algo.generate()
            main_algo.refine()

    if binary_mode:
        execute_login(main_algo.res_logger, main_algo.ens)
    else:
        execute_login(main_algo.res_logger, main_algo.container)

    writer = XMLProcessor(save_folder)

    if binary_mode:
        result = res_logger.get_results()
    else:
        result = res_logger.get_results()

    # Write both in one file, or write one in one file for multi
    if binary_mode:
        results = (result,)
    else:
        results = (result,)

    writer.write_all_variants(experiment, results=results, xml_names=xml_names)
    writer.save_in_one_file(experiment, file_name=file_name)

    # # Store the final preds in
    # timestamp = str(datetime.now()).replace(' ', '_')
    # timestamp = timestamp.replace(':', '_')
    # timestamp = timestamp.replace('.', '_')
    # fin_path = "C:/Users/vanya/OneDrive/Desktop/decolearn/decolearn/results/LPBoost_Fin_Preds/"
    # file = fin_path + 'fin_preds_' + timestamp + '.npy'
    # fin_preds_train = main_algo.ens.final_ens_pred_on_train
    # fin_preds_val = main_algo.ens.final_ens_pred_on_val
    # fin_preds_train_val = np.asarray([fin_preds_train, fin_preds_val], dtype=object)
    #
    # np.save(file=file, arr=fin_preds_train_val)

