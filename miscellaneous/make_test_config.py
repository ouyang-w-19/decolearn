import os
from utils.xml_processor import XMLProcessor


if __name__ == '__main__':

    mlp_mlp_exp = {'dataset': 'cifar100 - 20 cat.',
                   'n_sampl': 10000,
                   'nu'     : 0.2,
                   'bl_type': 'mlp',
                   'bl_type_ref': 'mlp',
                   'target_': 2,
                   'n_start': 2,    # Old: 5
                   'comb_me': 'av',
                   'bl_config':  {'inp_dim': (28, 28),
                                  'nn_arch': (1,),
                                  'le_rate': 0.001,
                                  'batch_s': 100,    # Old 50
                                  'n_episo': 1,
                                  'loss_fu': 'binary_crossentropy',
                                  'metric_': 0
                                  },
                   'bl_config_ref': {'inp_dim': (28, 28),
                                     'nn_arch': (1,),
                                     'le_rate': 0.001,
                                     'batch_s': 100,  # Old 50
                                     'n_episo': 1,
                                     'loss_fu': 'binary_crossentropy',
                                     'metric_': 0
                                     }
                   }

    mlp_cnn_exp = {'dataset': 'cifar100 - 20 cat.',
                   'n_sampl': 10000,
                   'nu'     : 0.2,
                   'bl_type': 'mlp',
                   'bl_type_ref': 'cnn',
                   'target_': 2,
                   'n_start': 2,    # Old: 5
                   'comb_me': 'av',
                   'bl_config':  {'inp_dim': (28, 28),
                                  'nn_arch': (1,),
                                  'le_rate': 0.001,
                                  'batch_s': 100,    # Old 50
                                  'n_episo': 1,
                                  'loss_fu': 'binary_crossentropy',
                                  'metric_': 0
                                  },
                   'bl_config_ref': {'inp_dim': (28, 28),
                                     'le_rate': 0.001,
                                     'loss_fu': 'binary_crossentropy',
                                     'batch_s': 100,
                                     'n_episo': 1,
                                     'metric_': 0,
                                     'n_filte': (4, 8, 4),
                                     's_filte': (2, 2),
                                     'activa': 'tanh',
                                     'multila': (4, 1)
                                     },
                   }

    mlp_tree_exp = {'dataset': 'cifar100 - 20 cat.',
                    'n_sampl': 10000,
                    'nu'     : 0.2,
                    'bl_type': 'mlp',
                    'bl_type_ref': 'tree',
                    'target_': 2,
                    'n_start': 2,    # Old: 5
                    'comb_me': 'av',
                    'bl_config':  {'inp_dim': (28, 28),
                                   'nn_arch': (1,),
                                   'le_rate': 0.001,
                                   'batch_s': 100,    # Old 50
                                   'n_episo': 1,
                                   'loss_fu': 'binary_crossentropy',
                                   'metric_': 0
                                   },
                    'bl_config_ref': {'max_dep': None,
                                      'max_fea': None,
                                      'max_lea': None,
                                      'min_imp': 0.0,
                                      'c_weigh': None
                                      },
                    }

    cnn_cnn_exp = {'dataset': 'cifar100 - 20 cat.',
                   'n_sampl': 10000,
                   'nu'     : 0.2,
                   'bl_type': 'cnn',
                   'bl_type_ref': 'cnn',
                   'target_': 2,
                   'n_start': 2,
                   'comb_me': 'av',
                   'bl_config':     {'inp_dim': (28, 28),
                                     'le_rate': 0.001,
                                     'loss_fu': 'binary_crossentropy',
                                     'batch_s': 100,
                                     'n_episo': 1,
                                     'metric_': 0,
                                     'n_filte': (4, 8, 4),
                                     's_filte': (2, 2),
                                     'activa': 'tanh',
                                     'multila': (5, 1)
                                     },
                   'bl_config_ref': {'inp_dim': (28, 28),
                                     'le_rate': 0.001,
                                     'loss_fu': 'binary_crossentropy',
                                     'batch_s': 100,
                                     'n_episo': 5,
                                     'metric_': 0,
                                     'n_filte': (4, 16, 8),
                                     's_filte': (2, 2),
                                     'activa': 'tanh',
                                     'multila': (8, 1)
                                     }
                   }

    cnn_mlp_exp = {'dataset': 'cifar100 - 20 cat.',
                   'n_sampl': 10000,
                   'nu'     : 0.2,
                   'bl_type': 'cnn',
                   'bl_type_ref': 'mlp',
                   'target_': 2,
                   'n_start': 1,
                   'comb_me': 'av',
                   'bl_config':     {'inp_dim': (28, 28),
                                     'le_rate': 0.001,
                                     'loss_fu': 'binary_crossentropy',
                                     'batch_s': 100,
                                     'n_episo': 5,
                                     'metric_': 0,
                                     'n_filte': (8, 8, 4),
                                     's_filte': (2, 2),
                                     'activa': 'tanh',
                                     'multila': (2, 1)
                                     },
                   'bl_config_ref': {'inp_dim': (28, 28),
                                     'nn_arch': (1,),
                                     'le_rate': 0.001,
                                     'batch_s': 100,  # Old 50
                                     'n_episo': 1,
                                     'loss_fu': 'binary_crossentropy',
                                     'metric_': 0
                                     },
                   }

    cnn_tree_exp = {'dataset': 'cifar100 - 20 cat.',
                    'n_sampl': 10000,
                    'nu'     : 0.2,
                    'bl_type': 'cnn',
                    'bl_type_ref': 'tree',
                    'target_': 2,
                    'n_start': 2,
                    'comb_me': 'av',
                    'bl_config':     {'inp_dim': (28, 28),
                                      'le_rate': 0.001,
                                      'loss_fu': 'binary_crossentropy',
                                      'batch_s': 100,
                                      'n_episo': 5,
                                      'metric_': 0,
                                      'n_filte': (4, 8, 4),
                                      's_filte': (2, 2),
                                      'activa': 'tanh',
                                      'multila': (4, 1)
                                      },
                    'bl_config_ref': {'max_dep': None,
                                      'max_fea': None,
                                      'max_lea': None,
                                      'min_imp': 0.0,
                                      'c_weigh': None
                                      },
                    }

    tree_tree_exp = {'dataset': 'cifar100 - 20 cat.',
                     'n_sampl': 10000,
                     'nu'     : 0.2,
                     'bl_type': 'tree',
                     'bl_type_ref': 'tree',
                     'target_': 2,
                     'n_start': 2,
                     'comb_me': 'av',
                     'bl_config':     {'max_dep': None,
                                       'max_fea': None,
                                       'max_lea': None,
                                       'min_imp': 0.0,
                                       'c_weigh': None
                                       },
                     'bl_config_ref': {'max_dep': None,
                                       'max_fea': None,
                                       'max_lea': None,
                                       'min_imp': 0.0,
                                       'c_weigh': None
                                       },

                     }

    tree_mlp_exp = {'dataset': 'cifar100 - 20 cat.',
                    'n_sampl': 10000,
                    'nu'     : 0.2,
                    'bl_type': 'tree',
                    'bl_type_ref': 'mlp',
                    'target_': 2,
                    'n_start': 2,
                    'comb_me': 'av',
                    'bl_config':     {'max_dep': None,
                                      'max_fea': None,
                                      'max_lea': None,
                                      'min_imp': 0.0,
                                      'c_weigh': None
                                       },
                    'bl_config_ref': {'inp_dim': (28, 28),
                                      'nn_arch': (1,),
                                      'le_rate': 0.001,
                                      'batch_s': 100,  # Old 50
                                      'n_episo': 1,
                                      'loss_fu': 'binary_crossentropy',
                                      'metric_': 0
                                      }

                    }

    tree_cnn_exp = {'dataset': 'cifar100 - 20 cat.',
                    'n_sampl': 10000,
                    'nu'     : 0.2,
                    'bl_type': 'tree',
                    'bl_type_ref': 'cnn',
                    'target_': 2,
                    'n_start': 2,
                    'comb_me': 'av',
                    'bl_config':     {'max_dep': None,
                                      'max_fea': None,
                                      'max_lea': None,
                                      'min_imp': 0.0,
                                      'c_weigh': None
                                      },
                    'bl_config_ref':     {'inp_dim': (28, 28),
                                          'le_rate': 0.001,
                                          'loss_fu': 'binary_crossentropy',
                                          'batch_s': 100,
                                          'n_episo': 1,
                                          'metric_': 0,
                                          'n_filte': (4, 8, 4),
                                          's_filte': (2, 2),
                                          'activa': 'tanh',
                                          'multila': (4, 1)
                                          },

                    }

    test_conf = [mlp_mlp_exp, mlp_cnn_exp, mlp_tree_exp, cnn_mlp_exp, cnn_cnn_exp, cnn_tree_exp,
                 tree_mlp_exp, tree_cnn_exp, tree_tree_exp]
    curr_dir = os.getcwd()
    main_dir = os.path.dirname(curr_dir)

    writer = XMLProcessor(curr_dir + '/algorithm/test_params.xml')

    for idx, config in enumerate(test_conf):
        writer.write_all(config, 'Test_config', f'Experiment {idx}')
    # writer.write(mlp_cnn_exp, 'SS2_MLP_CNN_5_Runs', 'Res: Train -> (1269.4, 746.4), Val -> (209.8, 130.6)')