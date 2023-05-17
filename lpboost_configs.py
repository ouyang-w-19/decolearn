"""
Contains all configuration dictinaries for LPBoost
"""


mlp_mlp_exp = {'dataset': 'cifar10',
               'n_sampl': 60000,
               'nu'     : 0.5,
               'bl_type': 'mlp',
               'bl_type_ref': 'mlp',
               'target_': 3,
               'n_start': 1,    # Old: 5
               'n_bags': 1,
               'comb_me': 'av',
               'binary_mode': True,
               'bl_config':  {'inp_dim': (32, 32, 3),
                              'nn_arch': (20, 15, 1),
                              'le_rate': 0.001,
                              'batch_s': 50,    # Old 50
                              'n_episo': 15,
                              'loss_fu': 'binary_crossentropy',
                              'activa': ('relu', 'relu', 'sigmoid'),
                              'metric_': 0
                              },
               'bl_config_ref': {'inp_dim': (32, 32, 3),
                                 'nn_arch': (20, 15, 1),
                                 'le_rate': 0.001,
                                 'batch_s': 50,  # Old 50
                                 'n_episo': 15,
                                 'loss_fu': 'binary_crossentropy',
                                 'activa': ('relu', 'relu', 'sigmoid'),
                                 'metric_': 0
                                 },
               'data': ''
               }

mlp_cnn_exp = {'dataset': 'cifar_10_t3_nt8',
               'n_sampl': 50000,
               'nu'     : 0.22,
               'bl_type': 'mlp',
               'bl_type_ref': 'cnn',
               'target_': 3,
               'n_start': 1,    # Old: 5
                'n_bags': 4,
               'comb_me': 'av',
               'binary_mode': True,
               'bl_config':  {'inp_dim': (32, 32, 3),
                              'nn_arch': (5, 4, 3, 2, 1),
                              'le_rate': 0.001,
                              'batch_s': 50,    # Old 50
                              'n_episo': 5,
                              'loss_fu': 'binary_crossentropy',
                              'activa': 'tanh',
                              'metric_': 0
                              },
               'bl_config_ref': {'archit' : 'vgg',
                                 'inp_dim': (32, 32, 3),
                                 'le_rate': 0.001,
                                 'loss_fu': 'binary_crossentropy',
                                 'batch_s': 50,
                                 'n_episo': 5,
                                 'metric_': 0,
                                 'n_filte': (16, 32, 16),
                                 's_filte': (2, 2),
                                 'pooling_filter': (2, 2),
                                 'activa': 'tanh',
                                 'multila': (8, 1)
                                 },
               'data': ''
               }

mlp_tree_exp = {'dataset': 'MNIST',
                'n_sampl': 60000,
                'nu'     : 0.2,
                'bl_type': 'mlp',
                'bl_type_ref': 'tree',
                'target_': 2,
                'n_start': 3,    # Old: 5
                'n_bags': 4,
                'comb_me': 'av',
                'binary_mode': True,
                'bl_config':  {'inp_dim': (28, 28),
                               'nn_arch': (3, 2, 1),
                               'le_rate': 0.001,
                               'batch_s': 50,    # Old 50
                               'n_episo': 5,
                               'loss_fu': 'binary_crossentropy',
                               'activa': 'tanh',
                               'metric_': 0
                               },
                'bl_config_ref': {'max_dep': None,
                                  'max_fea': None,
                                  'max_lea': None,
                                  'min_imp': 0.0,
                                  'c_weigh': None
                                  },
                'data': ''
                }

cnn_cnn_exp_binary = {'dataset': 'cifar10',
                      'n_sampl': 50000,
                      'nu'     : 0.5,
                      'bl_type': 'cnn',
                      'bl_type_ref': 'cnn',
                      'target_': 3,
                      'n_start': 1,
                      'n_bags': 1,
                      'comb_me': 'av',
                      'binary_mode': True,
                      'bl_config':     {'archit': 'vgg',
                                        'inp_dim': (32, 32, 3),
                                        'le_rate': 0.001,
                                        'loss_fu': 'binary_crossentropy',
                                        'batch_s': 25,    # Old: 25
                                        'n_episo': 10,     # 7
                                        'metric_': 0,
                                        'n_filte': (32, 32),   # Old: (64, 32, 32),
                                        's_filte': (2, 2),
                                        'pooling_filter': (2, 2),
                                        'activa': 'relu',
                                        'multila': (10, 1)   # (20, 15, 1)
                                        },
                      'bl_config_ref': {'archit': 'vggrelu',
                                        'inp_dim': (32, 32, 3),
                                        'le_rate': 0.001,
                                        'loss_fu': 'binary_crossentropy',
                                        'batch_s': 10,    # Old 50
                                        'n_episo': 10,     # 7
                                        'metric_': 0,
                                        'n_filte': (32, 32),    # (64, 32, 32)
                                        's_filte': (2, 2),
                                        'pooling_filter': (2, 2),
                                        'activa': 'relu',
                                        'multila': (10, 1)   # (20, 15, 1)
                                        },
                      'data': ''
                      }

cnn_enscnnvgg_exp_binary = {'dataset': 'cifar10',
                            'n_sampl': 50000,
                            'nu'     : 0.5,
                            'bl_type': 'cnn',
                            'bl_type_ref': 'enscnnvgg',
                            'target_': 3,
                            'n_start': 1,
                            'n_bags': 1,
                            'comb_me': 'av',
                            'binary_mode': True,
                            'bl_config':     {'archit': 'vgg',
                                              'inp_dim': (32, 32, 3),
                                              'le_rate': 0.001,
                                              'loss_fu': 'binary_crossentropy',
                                              'batch_s': 25,    # Old: 25
                                              'n_episo': 5,     # 7
                                              'metric_': 0,
                                              'n_filte': (8,),   # Old: (64, 32, 32),
                                              's_filte': (2, 2),
                                              'pooling_filter': (2, 2),
                                              'activa': 'relu',
                                              'multila': (15, 1)
                                              },
                            'bl_config_ref': {'inp_dim': (32, 32, 3),
                                              'le_rate': 0.001,
                                              'loss_fu': 'binary_crossentropy',
                                              'batch_s': 1,    # Old 50
                                              'n_episo': 7,     # 7
                                              'metric_': 0,
                                              'n_filte': (8,),    # (64, 32, 32)
                                              's_filte': (2, 2),
                                              'pooling_filter': (2, 2),
                                              # For CNN and MLP
                                              'activa': ('relu', 'relu', 'relu', 'sigmoid'),
                                              'multila': (15, 1)
                                              },
                            'data': ''
                            }

enscnnvgg_enscnnvgg_exp_binary = {'dataset': 'cifar10',
                                  'n_sampl': 50000,
                                  'nu'     : 0.5,
                                  'bl_type': 'cnn',
                                  'bl_type_ref': 'cnn',
                                  'target_': 3,
                                  'n_start': 1,
                                  'n_bags': 1,
                                  'comb_me': 'av',
                                  'binary_mode': True,
                                  'bl_config':     {'inp_dim': (32, 32, 3),
                                                    'le_rate': 0.001,
                                                    'loss_fu': 'binary_crossentropy',
                                                    'batch_s': 25,    # Old: 25
                                                    'n_episo': 3,     # 7
                                                    'metric_': 0,
                                                    'n_filte': (4,),   # Old: (64, 32, 32),
                                                    's_filte': (2, 2),
                                                    'pooling_filter': (2, 2),
                                                    'activa': 'relu',
                                                    'multila': (15, 1)
                                                    },
                                  'bl_config_ref': {'inp_dim': (32, 32, 3),
                                                    'le_rate': 0.001,
                                                    'loss_fu': 'binary_crossentropy',
                                                    'batch_s': 50,    # Old 50
                                                    'n_episo': 3,     # 7
                                                    'metric_': 0,
                                                    'n_filte': (4,),    # (64, 32, 32)
                                                    's_filte': (2, 2),
                                                    'pooling_filter': (2, 2),
                                                    'activa': 'relu',
                                                    'multila': (15, 1)
                                                    },
                                  'data': ''
                                  }

cnn_cnn_exp_multi = {'dataset': 'cifar10',
                      'n_sampl': 50000,
                      'nu'     : 0.5,
                      'bl_type': 'cnn',
                      'bl_type_ref': 'cnn',
                      'target_': 'all',
                      'n_start': 3,     # 3
                      'n_bags': 1,
                      'comb_me': 'av',
                      'binary_mode': False,
                      'bootstrapping': False,
                      'bl_config':     {'archit': 'vgg',
                                        'inp_dim': (32, 32, 3),
                                        # 'inp_dim': (28, 28),
                                        'le_rate': 0.001,
                                        'loss_fu': 'sparse_categorical_crossentropy',
                                        'batch_s': 50,    # Old: 25
                                        'n_episo': 1,     # 5
                                        'metric_': 1,
                                        'n_filte': (16,),   # Old: (64, 32, 32),
                                        's_filte': (2, 2),
                                        'pooling_filter': (2, 2),
                                        'activa': 'relu',
                                        'multila': (15, 10)     # Old: (20, 15, 10)
                                        },
                      'bl_config_ref': {'archit': 'vgg',
                                        'inp_dim': (32, 32, 3),
                                        # 'inp_dim': (28, 28),
                                        'le_rate': 0.001,
                                        'loss_fu': 'sparse_categorical_crossentropy',
                                        'batch_s': 50,    # Old 50
                                        'n_episo': 1,     # 5
                                        'metric_': 1,
                                        'n_filte': (16,),    # Old: (64, 32, 32)
                                        's_filte': (2, 2),
                                        'pooling_filter': (2, 2),
                                        'activa': 'relu',
                                        'multila': (15, 10)     # Old: (20, 15, 10)
                                        },
                      'data': ''
                      }

cnn_mlp_exp = {'dataset': 'cifar10_t3_nt8',
               'n_sampl': 50000,
               'nu'     : 0.2,
               'bl_type': 'cnn',
               'bl_type_ref': 'mlp',
               'target_': 3,
               'n_start': 5,
               'n_bags': 4,
               'comb_me': 'av',
               'binary_mode': True,
               'bl_config':    {'archit': 'vgg',
                                'inp_dim': (32, 32, 3),
                                'le_rate': 0.001,
                                'loss_fu': 'binary_crossentropy',
                                'batch_s': 50,
                                'n_episo': 5,
                                'metric_': 0,
                                'n_filte': (16, 32, 16),
                                's_filte': (2, 2),
                                'pooling_filter': (2, 2),
                                'activa': 'tanh',
                                'multila': (8, 1)
                                },
               'bl_config_ref': {'inp_dim': (32, 32, 3),
                                 'nn_arch': (3, 2, 1),
                                 'le_rate': 0.001,
                                 'batch_s': 50,  # Old 50
                                 'n_episo': 5,
                                 'loss_fu': 'binary_crossentropy',
                                 'activa': 'tanh',
                                 'metric_': 0
                                 },
               'data': ''
               }

cnn_tree_exp = {'dataset': 'cifar10_t3_nt8',
                'n_sampl': 50000,
                'nu'     : 0.2,
                'bl_type': 'cnn',
                'bl_type_ref': 'tree',
                'target_': 3,
                'n_start': 1,
                'n_bags': 4,
                'comb_me': 'av',
                'binary_mode': True,
                'bl_config': {'archit': 'vgg',
                              'inp_dim': (32, 32, 3),
                              'le_rate': 0.001,
                              'loss_fu': 'binary_crossentropy',
                              'batch_s': 50,
                              'n_episo': 5,
                              'metric_': 0,
                              'n_filte': (16, 32, 16),
                              's_filte': (2, 2),
                              'pooling_filter': (2, 2),
                              'activa': 'tanh',
                              'multila': (8, 1)
                              },
                'bl_config_ref': {'max_dep': None,
                                  'max_fea': None,
                                  'max_lea': None,
                                  'min_imp': 0.0,
                                  'c_weigh': None
                                  },
                'data': ''
                }

tree_tree_exp = {'dataset': 'cifar10',
                 'n_sampl': 50000,
                 'nu'     : 0.5,
                 'bl_type': 'tree',
                 'bl_type_ref': 'tree',
                 'target_': 3,
                 'n_start': 1,
                 'n_bags': 1,
                 'comb_me': 'av',
                 'binary_mode': True,
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
                 'data': ''
                 }

tree_mlp_exp = {'dataset': 'cifar10_t3_nt8',
                'n_sampl': 50000,
                'nu'     : 0.2,
                'bl_type': 'tree',
                'bl_type_ref': 'mlp',
                'target_': 3,
                'n_start': 1,
                'n_bags': 4,
                'comb_me': 'av',
                'binary_mode': True,
                'bl_config':     {'max_dep': 1,
                                  'max_fea': None,
                                  'max_lea': None,
                                  'min_imp': 0.0,
                                  'c_weigh': None
                                  },
                'bl_config_ref': {'inp_dim': (32, 32, 3),
                                  'nn_arch': (5, 4, 3, 2, 1),
                                  'le_rate': 0.001,
                                  'batch_s': 50,  # Old 50
                                  'n_episo': 5,
                                  'loss_fu': 'binary_crossentropy',
                                  'activa': 'tanh',
                                  'metric_': 0
                                  },
                'data': ''

                }

tree_cnn_exp = {'dataset': 'cifar10_t3_nt8',
                'n_sampl': 50000,
                'nu'     : 0.2,
                'bl_type': 'tree',
                'bl_type_ref': 'cnn',
                'target_': 3,
                'n_start': 1,
                'n_bags': 4,
                'comb_me': 'av',
                'binary_mode': True,
                'bl_config':     {'max_dep': None,
                                  'max_fea': None,
                                  'max_lea': None,
                                  'min_imp': 0.0,
                                  'c_weigh': None
                                  },
                'bl_config_ref':     {'inp_dim': (32, 32, 3),
                                      'le_rate': 0.001,
                                      'loss_fu': 'binary_crossentropy',
                                      'batch_s': 50,
                                      'n_episo': 5,
                                      'metric_': 0,
                                      'n_filte': (16, 32, 16),
                                      's_filte': (2, 2),
                                      'pooling_filter': (2, 2),
                                      'activa': 'tanh',
                                      'multila': (8, 1)
                                      },
                'data': ''

                }

cnn_xgboost_exp = {'dataset': 'cifar_10_t3_nt2',
                   'n_sampl': 50000,
                   'nu'     : 0.2,
                   'bl_type': 'cnn',
                   'bl_type_ref': 'xgboost',
                   'target_': 3,
                   'n_start': 5,
                   'n_bags': 4,
                   'comb_me': 'av',
                   'binary_mode': True,
                   'bl_config':     {'archit': 'vgg',
                                     'inp_dim': (32, 32, 3),
                                     'le_rate': 0.001,
                                     'loss_fu': 'binary_crossentropy',
                                     'batch_s': 50,
                                     'n_episo': 5,     # 5
                                     'metric_': 0,
                                     'n_filte': (32, 32, 16),
                                     's_filte': (2, 2),
                                     'pooling_filter': (2, 2),
                                     'activa': 'tanh',
                                     'multila': (8, 1)
                                     },
                   'bl_config_ref': {'inp_dim': (32, 32, 3),
                                     'le_rate': 0.05,
                                     'n_estimators': 10,
                                     'sub_sample': 1,     # <1.0 leads to -variance +bias, old: 1
                                     'min_samples_split': 2,
                                     'max_depth': 3,
                                     'tol': 0.0001
                                     },
               'data': ''
               }


xgboost_cnn_exp = {'dataset': 'cifar_10_t3_nt2',
                   'n_sampl': 50000,
                   'nu'     : 0.2,
                   'bl_type': 'xgboost',
                   'bl_type_ref': 'cnn',
                   'target_': 3,
                   'n_start': 1,
                   'n_bags': 4,
                   'comb_me': 'av',
                   'binary_mode': True,
                   'bl_config': {'inp_dim': (32, 32, 3),
                                 'le_rate': 0.1,
                                 'n_estimators': 10,
                                 'sub_sample': 1.0,  # <1.0 leads to -variance +bias
                                 'min_samples_split': 2,
                                 'max_depth': 3,
                                 'tol': 0.0001
                                 },
                   'bl_config_ref': {'archit': 'vgg',
                                     'inp_dim': (32, 32, 3),
                                     'le_rate': 0.001,
                                     'loss_fu': 'binary_crossentropy',
                                     'batch_s': 50,
                                     'n_episo': 5,     # 8
                                     'metric_': 0,
                                     'n_filte': (32, 32, 16),
                                     's_filte': (2, 2),
                                     'pooling_filter': (2, 2),
                                     'activa': 'tanh',
                                     'multila': (8, 1)
                                     },
                   'data': ''
                   }


xgboost_xgboost_exp = {'dataset': 'cifar_10_t3_nt2',
                       'n_sampl': 50000,
                       'nu'     : 0.2,
                       'bl_type': 'xgboost',
                       'bl_type_ref': 'xgboost',
                       'target_': 3,
                       'n_start': 1,
                       'n_bags': 4,
                       'comb_me': 'av',
                       'binary_mode': True,
                       'bl_config': {'inp_dim': (32, 32, 3),
                                     'le_rate': 0.1,
                                     'n_estimators': 10,
                                     'sub_sample': 1.0,  # <1.0 leads to -variance +bias
                                     'min_samples_split': 2,
                                     'max_depth': 3,
                                     'tol': 0.0001
                                     },
                       'bl_config_ref': {'inp_dim': (32, 32, 3),
                                         'le_rate': 0.1,
                                         'n_estimators': 10,
                                         'sub_sample': 1.0,  # <1.0 leads to -variance +bias
                                         'min_samples_split': 2,
                                         'max_depth': 3,
                                         'tol': 0.0001
                                         },
                   'data': ''
                   }

##############################################################################################################
mlp_mlp_test = {'dataset': 'mnist',
                'n_sampl': 10000,
                'nu'     : 0.8,
                'bl_type': 'mlp',
                'bl_type_ref': 'mlp',
                'target_': 'all',
                'n_start': 1,    # Old: 5
                'n_bags': 4,
                'comb_me': 'av',
                'binary_mode': False,
                'bl_config':  {'inp_dim': (28, 28),
                               'nn_arch': (30, 20, 10),
                               'le_rate': 0.001,
                               'batch_s': 50,    # Old 50
                               'n_episo': 6,
                               'loss_fu': 'sparse_categorical_crossentropy',
                               'activa': 'relu',
                               'metric_': 1
                               },
                'bl_config_ref': {'inp_dim': (28, 28),
                                  'nn_arch': (30, 20, 10),
                                  'le_rate': 0.001,
                                  'batch_s': 50,  # Old 50
                                  'n_episo': 6,
                                  'loss_fu': 'sparse_categorical_crossentropy',
                                  'activa': 'relu',
                                  'metric_': 1
                                  },
                'data': ''
                }


mlp_mlp_test_binary = {'dataset': 'mnist',
                       'n_sampl': 10000,
                       'nu'     : 0.2,
                       'bl_type': 'mlp',
                       'bl_type_ref': 'mlp',
                       'target_': 3,
                       'n_start': 1,    # Old: 5
                       'n_bags': 4,
                       'comb_me': 'av',
                       'binary_mode': True,
                       'bl_config':  {'inp_dim': (28, 28),
                                      'nn_arch': (3, 2, 1),
                                      'le_rate': 0.001,
                                      'batch_s': 50,    # Old 50
                                      'n_episo': 6,
                                      'loss_fu': 'binary_crossentropy',
                                      'activa': 'tanh',
                                      'metric_': 0
                                      },
                       'bl_config_ref': {'inp_dim': (28, 28),
                                         'nn_arch': (3, 2, 1),
                                         'le_rate': 0.001,
                                         'batch_s': 50,  # Old 50
                                         'n_episo': 6,
                                         'loss_fu': 'binary_crossentropy',
                                         'activa': 'tanh',
                                         'metric_': 0
                                         },
                       'data': ''
                       }