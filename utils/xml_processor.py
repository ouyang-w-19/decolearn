import xml.etree.ElementTree as ET
from datetime import datetime


class XMLProcessor:
    def __init__(self,
                 save_folder,
                 meta_params=('bl_config', 'bl_config_ref', 'dataset', 'n_sampl', 'nu', 'bl_type',
                              'bl_type_ref', 'target_', 'n_start', 'n_bags', 'comb_me', 'binary_mode', 'data'),
                 iteration_nodes=('mp_value', 'bl_acc_spec', 'bl_acc_or_train', 'bl_acc_or_val',
                                  'miss_train', 'miss_val', 'avg_div', 'alpha'),
                 mlp_bl_params=('inp_dim', 'nn_arch', 'le_rate', 'batch_s', 'n_episo', 'loss_fu', 'activa', 'metric_'),
                 cnn_bl_params=('archit', 'inp_dim', 'le_rate', 'loss_fu', 'batch_s', 'n_episo', 'metric_', 'n_filte',
                                's_filte', 'pooling_filter', 'activa', 'multila'),
                 ens_cnnvgg=('inp_dim', 'le_rate', 'loss_fu', 'batch_s', 'n_episo', 'metric_', 'n_filte',
                                's_filte', 'pooling_filter', 'activa', 'multila'),
                 tree_bl_params=('max_dep', 'max_fea', 'max_lea', 'min_imp', 'c_weigh'),
                 deep_meta_nodes=('bl_config', 'bl_config_ref')
                 ):
        """
        - Saves and loads (configs) and (configs+data)
        :param save_folder: Folder in which XML object will be stored
        :param meta_params: Nodes to be created directly under root node
        :param iteration_nodes: Nodes to be created in each iteration node under phase node
        :param mlp_bl_params: Configurations for MLP BL
        :param cnn_bl_params: Configurations for CNN BL
        :param tree_bl_params: Configurations for tree (CART) algorithm
        :param deep_meta_nodes: Stores in att. self.deep_nodes to know which node texts to skip when creating meta-nodes
        """
        # Node Names #
        self._meta_params = meta_params
        self.deep_nodes = deep_meta_nodes

        # Sub-Node Config Names #
        self._mlp_bl_params = mlp_bl_params
        self._cnn_bl_params = cnn_bl_params
        self._ens_cnnvgg = ens_cnnvgg
        self._tree_bl_params = tree_bl_params
        self._xg_boost_params = ['inp_dim', 'le_rate', 'n_estimators', 'sub_sample', 'min_samples_split', 'max_depth',
                                 'tol']
        self._data = []

        # Sub-Node Data
        self._data_sub_nodes = ['alt_ini_mp_ens_train', 'alt_ini_mp_ens_val',
                                'bagging_phase', 'boosting_phase']

        self._data_time_sub2_node = ['perf_time', 'perf_time__ref']

        # Sub-Sub-Sub-Nodes
        self._iterations = iteration_nodes

        self._xml_path = None
        self._save_folder = save_folder
        self._root_node = ET.Element('Experiment_Log')

        # self._previous_experiments = self.load_params()

        # self._xml = ET.parse(xml_path)
        # self._xml_path = xml_path
        # self._root_node = self._xml.getroot()

    """
    Wrappers
    """
    def save_in_one_file(self, exp: dict, file_name):
        type_gen = exp['bl_type'].lower()
        type_ref = exp['bl_type_ref'].lower()
        exp_type = type_gen + '_' + type_ref
        dataset = exp['dataset'].lower()
        self.save_xml(self._root_node, exp_type=exp_type, dataset=dataset, file_name=file_name)

    def save_xml(self, xml_obj, exp_type: str, dataset='', file_name=''):
        # Make timestamp
        timestamp = str(datetime.now()).replace(' ', '_')
        timestamp = timestamp.replace(':', '_')
        timestamp = timestamp.replace('.', '_')
        # if not dataset:
        #     dataset = ''
        if file_name:
            name = exp_type + '__' + dataset + '__' + file_name + '__' + timestamp + '.xml'
        else:
            name = exp_type + '__' + dataset + '__' + timestamp + '.xml'
        path_name = self._save_folder + name

        mydata = ET.tostring(xml_obj).decode('utf-8')
        # if os.path.isfile(self._xml_path):
        #     os.remove(self._xml_path)
        with open(path_name, 'w') as myfile:
            myfile.write(mydata)

        print('Saved to XML file')

    def write_all_variants(self, exp: dict, results: tuple, xml_names: tuple):
        for result, xml_name in zip(results, xml_names):
            self.write_all(exp=exp, res_data=result, xml_name=xml_name)
        return 0
    """
    Wrappers
    """

    """
    Helper Functions
    """
    @staticmethod
    def create_subnode(node, sub_node_name: str, sub_node_value):
        """
        - Creates a subnote 'sub_node_name', under 'node' and fills it with 'sub_node_values'
        :param node: Parent node
        :param sub_node_name: Sub-node to be created
        :param sub_node_value: Value of subnode set with subnode.text
        :return: Subnode as independent object
        """
        if type(sub_node_value).__name__ == 'ndarray':
            node_value = str(sub_node_value.tolist())
        else:
            node_value = sub_node_value
        subnode = ET.SubElement(node, sub_node_name)
        subnode.text = node_value

        return subnode

    @staticmethod
    def create_subnode_with_skip(node, sub_node_name: str, sub_node_value, skip_nodes):
        if type(sub_node_value).__name__ == 'ndarray':
            sub_node_value = str(sub_node_value.tolist())
        else:
            sub_node_value = sub_node_value
        subnode = ET.SubElement(node, sub_node_name)
        subnode.text = sub_node_value if sub_node_name not in skip_nodes else ''

        return subnode
    """
    /Helper Functions
    """

    def clear_all_nodes(self):
        for node in self._root_node:
            self._root_node.remove(node)

    def load_configs(self, xml_obj_path):
        root = ET.parse(xml_obj_path)
        bl_config = []
        bl_config_ref = []
        rest = []

        d_1 = dict()
        d_config_2 = dict()
        d_ref_config_2 = dict()

        for exp_idx, experiments in enumerate(root.iterfind('Configuration')):
            for sub_node_gen in experiments.iterfind('bl_config'):
                for element in sub_node_gen:
                    string_val = element.text
                    try:
                        d_config_2[element.tag] = eval(string_val)
                    except:
                        d_config_2[element.tag] = string_val

                bl_config.append(d_config_2)
                d_config_2 = {}

            for sub_node_ref in experiments.iterfind('bl_config_ref'):
                for element in sub_node_ref:
                    string_val = element.text
                    try:
                        d_ref_config_2[element.tag] = eval(string_val)
                    except:
                        d_ref_config_2[element.tag] = string_val

                bl_config_ref.append(d_ref_config_2)
                d_ref_config_2 = {}

            for sub_node_rest in experiments[2:]:
                sval = sub_node_rest.text
                try:
                    d_1[sub_node_rest.tag] = eval(sval)
                except:
                    d_1[sub_node_rest.tag] = sval

            d_1['bl_config'] = bl_config[exp_idx]
            d_1['bl_config_ref'] = bl_config_ref[exp_idx]
            rest.append(d_1)
            d_1 = {}

        return rest

    def load_all(self, xml_path, n_exps=None):
        self._xml_path = xml_path
        root = ET.parse(self._xml_path)
        bl_config = []
        bl_config_ref = []
        data = []
        rest = []

        d_1 = dict()
        d_config_2 = dict()
        d_ref_config_2 = dict()
        d_data_2 = dict()
        d_iter_bagging_3 = dict()
        d_iter_boosting_3 = dict()
        d_iter_val_4 = dict()

        for exp_idx, experiments in enumerate(root.iterfind('experiment')):
            for sub_node_gen in experiments.iterfind('bl_config'):
                for element in sub_node_gen:
                    string_val = element.text
                    try:
                        d_config_2[element.tag] = eval(string_val)
                    except:
                        d_config_2[element.tag] = string_val
                bl_config.append(d_config_2)
                d_config_2 = {}

            for sub_node_ref in experiments.iterfind('bl_config_ref'):
                for element in sub_node_ref:
                    string_val = element.text
                    try:
                        d_ref_config_2[element.tag] = eval(string_val)
                    except:
                        d_ref_config_2[element.tag] = string_val

                bl_config_ref.append(d_ref_config_2)
                d_ref_config_2 = {}

            # Load data to hierarchical dictionary, made of sub-dics
            data_length = len(experiments.find('data'))

            for sub_node_data_2 in experiments.iterfind('data'):
                for element in sub_node_data_2:
                    string_val = element.text
                    try:
                        d_data_2[element.tag] = eval(string_val)
                    except:
                        d_data_2[element.tag] = string_val

                # Find all iters in bagging/boosting phase and store in tmp lvl4 dictionary
                for sub3_iter_bagging in experiments.find('data').find('bagging_phase'):
                    # Lvl4: Values
                    for sub4_val_bagging in sub3_iter_bagging:
                        string_val = sub4_val_bagging.text
                        try:
                            d_iter_val_4[sub4_val_bagging.tag] = eval(string_val)
                        except:
                            d_iter_val_4[sub4_val_bagging.tag] = string_val
                    d_iter_bagging_3[sub3_iter_bagging.tag] = d_iter_val_4
                    d_iter_val_4 = {}

                # Find all phases in one data node and store th lvl3 nodes in a dict
                for sub3_iter_boosting in experiments.find('data').find('boosting_phase'):
                    # Lvl4: Values
                    for sub4_val_boosting in sub3_iter_boosting:
                        string_val = sub4_val_boosting.text
                        try:
                            d_iter_val_4[sub4_val_boosting.tag] = eval(string_val)
                        except:
                            d_iter_val_4[sub4_val_boosting.tag] = string_val
                    d_iter_boosting_3[sub3_iter_boosting.tag] = d_iter_val_4
                    d_iter_val_4 = {}

                d_data_2['bagging_phase'] = d_iter_bagging_3
                d_data_2['boosting_phase'] = d_iter_boosting_3

                data.append(d_data_2)
                d_data_2 = {}

            for sub_node_rest in experiments[2:]:
                sval = sub_node_rest.text
                try:
                    d_1[sub_node_rest.tag] = eval(sval)
                except:
                    d_1[sub_node_rest.tag] = sval

            # d_1['bl_config'] = bl_config[exp_idx]
            # d_1['bl_config_ref'] = bl_config_ref[exp_idx]
            # d_1['data'] = data[exp_idx]
            d_1['bl_config'], = bl_config
            d_1['bl_config_ref'], = bl_config_ref
            d_1['data'], = data

            rest.append(d_1)

            d_1 = {}
            bl_config = []
            bl_config_ref = []
            d_iter_bagging_3 = {}
            d_iter_boosting_3 = {}
            data = []

        # rest, = rest
        if n_exps:
            rest = rest[:n_exps]
        else:
            rest = rest[0]

        return rest

    def write_configs(self, experiments: list, previous_exp_path=None):
        """
        Writes the config data only to a XML file
        :param experiments:
        :type experiments:
        :param previous_exp_path:
        :type previous_exp_path:
        :return:
        :rtype:
        """

        params = self._meta_params[:-1]
        if previous_exp_path:
            root_node_config = self.load_configs(previous_exp_path)
        else:
            root_node_config = ET.Element('Configurations')

        file_name = 'Config'

        for experiment in experiments:
            bl_gen = experiment['bl_type'].lower()
            bl_ref = experiment['bl_type_ref'].lower()
            node_objs = []
            exp_i = ET.SubElement(root_node_config, 'Configuration')

            for node in params:
                node_obj = ET.SubElement(exp_i, node)
                node_obj.text = str(experiment[node]) if node not in self.deep_nodes else ''
                node_objs.append(node_obj)

            if bl_gen == 'mlp':
                node_type_gen = self._mlp_bl_params
            elif bl_gen == 'cnn':
                node_type_gen = self._cnn_bl_params
            elif bl_gen == 'enscnnvgg':
                node_type_gen = self._ens_cnnvgg
            elif bl_gen == 'tree':
                node_type_gen = self._tree_bl_params
            elif bl_gen == 'xgboost':
                node_type_gen = self._xg_boost_params
            else:
                print(f'Error! {bl_gen} type not known')
                return 1

            if bl_ref == 'mlp':
                node_type_ref = self._mlp_bl_params
            elif bl_ref == 'cnn':
                node_type_ref = self._cnn_bl_params
            elif bl_ref == 'enscnnvgg':
                node_type_ref = self._ens_cnnvgg
            elif bl_ref == 'tree':
                node_type_ref = self._tree_bl_params
            elif bl_ref == 'xgboost':
                node_type_ref = self._xg_boost_params
            else:
                print(f'Error! {bl_ref} type not known')
                return 1

            # Write the configuration in the node as a string
            for sub_node_gen in node_type_gen:
                sub_node_obj = ET.SubElement(node_objs[0], sub_node_gen)
                # Double Key: Access element of dic within dic
                sub_node_obj.text = str(experiment['bl_config'][sub_node_gen])

            for sub_node_ref in node_type_ref:
                sub_node_obj = ET.SubElement(node_objs[1], sub_node_ref)
                # Double Key: Access element of dic within dic
                sub_node_obj.text = str(experiment['bl_config_ref'][sub_node_ref])

        self.save_xml(root_node_config, exp_type=file_name)

    def write_all(self, exp: dict, res_data: tuple, xml_name='', save=False):
        """Result data format: mp_val, miss_train, miss_val, div_datapoints"""
        iter_i = 0
        # Configuration Saving
        bl_gen = exp['bl_type'].lower()
        bl_ref = exp['bl_type_ref'].lower()
        exp_i = ET.SubElement(self._root_node, 'experiment')
        exp_i.set('Variant', xml_name)
        node_objs_meta = []

        # Strings in File Name
        exp_type = bl_gen + '_' + bl_ref
        dataset = exp['dataset'].lower()

        # Write the meta-parameters
        for node in self._meta_params:
            node_obj = ET.SubElement(exp_i, node)
            node_obj.text = str(exp[node]) if node not in self.deep_nodes else ''
            node_objs_meta.append(node_obj)

        # Set to Correct BL Format for Gen. and Ref. Phase
        if bl_gen == 'mlp':
            node_type_gen = self._mlp_bl_params
        elif bl_gen == 'cnn':
            node_type_gen = self._cnn_bl_params
        elif bl_gen == 'enscnnvgg':
            node_type_gen = self._ens_cnnvgg
        elif bl_gen == 'tree':
            node_type_gen = self._tree_bl_params
        elif bl_gen == 'xgboost':
            node_type_gen = self._xg_boost_params
        else:
            raise Exception(f'Error! {bl_gen} type not known')

        if bl_ref == 'mlp':
            node_type_ref = self._mlp_bl_params
        elif bl_ref == 'cnn':
            node_type_ref = self._cnn_bl_params
        elif bl_ref == 'enscnnvgg':
            node_type_ref = self._ens_cnnvgg
        elif bl_ref == 'tree':
            node_type_ref = self._tree_bl_params
        elif bl_ref == 'xgboost':
            node_type_ref = self._xg_boost_params
        else:
            raise Exception(f'Error! {bl_gen} type not known')

        # Write the Configuration in the Node as a String
        for sub_node_gen in node_type_gen:
            sub_node_obj = ET.SubElement(node_objs_meta[0], sub_node_gen)
            # Double Key: Access element of dic within dic
            sub_node_obj.text = str(exp['bl_config'][sub_node_gen])

        for sub_node_ref in node_type_ref:
            sub_node_obj = ET.SubElement(node_objs_meta[1], sub_node_ref)
            # Double Key: Access element of dic within dic
            sub_node_obj.text = str(exp['bl_config_ref'][sub_node_ref])

        # Data Writing
        mp_val, miss_train, miss_val, avg_div, div_datapoints, alpha, mp_ini_train, mp_ini_val, gen_end, \
            bl_accuracies, average_bl_accuracy, perf_time_val, ref_bl_column_non_r, ref_column_r, \
            u_history, ini_preds_train, ini_preds_val, ref_preds_train, ref_preds_val, margin, multiple_y_hist, \
            bl_accuracies_on_original_train, bl_accuracies_on_original_val, \
            = res_data
        # Note: Order is important in iter_values
        iter_values = [mp_val, bl_accuracies, bl_accuracies_on_original_train, bl_accuracies_on_original_val,
                       miss_train, miss_val, avg_div, alpha, multiple_y_hist, margin, u_history,
                       ini_preds_train, ini_preds_val, ref_preds_train, ref_preds_val, ref_bl_column_non_r,
                       ref_column_r, div_datapoints]
        mp_ini_scores = mp_ini_train, mp_ini_val

        # Write data sub-nodes
        data_sub_nodes = []

        # Performance time in subnode 'data'
        data_node_perf_time = ET.SubElement(node_objs_meta[-1], 'perf_time')
        data_node_perf_time.text = str(perf_time_val)
        data_sub_nodes.append(data_node_perf_time)

        # Average BL accuracy in subnode 'data'
        data_node_avg_bl = ET.SubElement(node_objs_meta[-1], 'avg_bl_acc')
        data_node_avg_bl.text = str(average_bl_accuracy)
        data_sub_nodes.append(data_node_perf_time)

        for idx, data_sub_node in enumerate(self._data_sub_nodes):
            data_node_obj = ET.SubElement(node_objs_meta[-1], data_sub_node)
            data_node_obj.text = str(mp_ini_scores[idx]) if data_sub_node not in ['bagging_phase', 'boosting_phase'] \
                else ''
            data_sub_nodes.append(data_node_obj)

        # Create the iter sub2-nodes in the bagging-phase sub-nodes
        bagging_iter_length = gen_end
        bagging_iter_objs = []
        for i in range(bagging_iter_length):
            iter_name = 'iter_' + str(iter_i)
            bagging_iter = ET.SubElement(data_sub_nodes[-2], iter_name)
            bagging_iter_objs.append(bagging_iter)
            iter_i += 1

        # Iterate over bagging_iter_i and add values in sub3-nodes
        for idx, iter in enumerate(bagging_iter_objs):
            for j, bagging_iter_val in enumerate(self._iterations):
                sub2_node_obj = ET.SubElement(iter, bagging_iter_val)
                sub2_node_obj.text = str(iter_values[j][idx]) if iter_values[j] is not iter_values[0] else ''

            # Long numpy arrays converted to a list in order to prevent replacement with '...'
            sub2_node_margin_i = self.create_subnode(iter, 'margin', iter_values[-9][idx])
            sub2_node_u_i = self.create_subnode(iter, 'u_i', iter_values[-8][idx])
            sub2_node_ini_preds_train_i = self.create_subnode(iter, 'ini_preds_train', iter_values[-7][idx])
            sub2_node_ini_preds_val_i = self.create_subnode(iter, 'ini_preds_val', iter_values[-6][idx])

        # Edge case: iter_{ref} = 0, keep the structure
        # ref_iter_length = len(mp_val)
        ref_iter_length = len(alpha) - gen_end
        if ref_iter_length == 0:
            ref_iter_length = 1
            iter_values = [' '] * len(iter_values)
        # / Edge Case

        # Create the iter sub2-nodes in the boosting-phase sub-nodes
        boosting_iter_objs = []
        for i in range(ref_iter_length):
            iter_name = 'iter_' + str(iter_i)
            sub_node_iter_obj = ET.SubElement(data_sub_nodes[-1], iter_name)
            boosting_iter_objs.append(sub_node_iter_obj)
            iter_i += 1

        # Iterate over refinement_iter_i and add values in sub3-nodes
        for idx, iter in enumerate(boosting_iter_objs):
            for j, sub2_node_iter in enumerate(self._iterations):
                sub2_node_obj = ET.SubElement(iter, sub2_node_iter)
                sub2_node_obj.text = str(iter_values[j][idx+bagging_iter_length]) if iter_values[j] is not \
                    iter_values[0] else str(iter_values[j][idx])

        # Long numpy arrays must first be converted to a list in order to prevent replacement with '...'
            node_value_raw = iter_values[-1][idx]
            if type(node_value_raw).__name__ == 'ndarray':
                node_value = str(node_value_raw.tolist())
            else:
                node_value = node_value_raw
            sub2_node_div_datapoint = ET.SubElement(iter, 'div_datapoints')
            sub2_node_div_datapoint.text = node_value

            # Preds on R only
            node_value_raw_r = iter_values[-2][idx]
            if type(node_value_raw_r).__name__ == 'ndarray':
                node_value = str(node_value_raw_r.tolist())
            else:
                node_value = node_value_raw_r
            sub2_node_ref_column_r = ET.SubElement(iter, 'ref_column_r')
            sub2_node_ref_column_r.text = node_value

            # Preds on T\R must be stored individually, too
            node_value_raw_non_r = iter_values[-3][idx]
            if type(node_value_raw_non_r).__name__ == 'ndarray':
                node_value = str(node_value_raw_non_r.tolist())
            else:
                node_value = node_value_raw_non_r
            sub2_node_ref_column_non_r = ET.SubElement(iter, 'ref_column_non_r')
            sub2_node_ref_column_non_r.text = node_value

            sub2_node_multiple_y_indicator_i = self.create_subnode(iter, 'y_balance', iter_values[-10][idx])
            sub2_node_margin_i = self.create_subnode(iter, 'margin', iter_values[-9][idx])
            sub2_node_u_i = self.create_subnode(iter, 'u_i', iter_values[-8][gen_end+idx])
            sub2_node_ref_preds_train_i = self.create_subnode(iter, 'ref_preds_train', iter_values[-5][idx])
            sub2_node_ref_preds_val_i = self.create_subnode(iter, 'ref_preds_val', iter_values[-4][idx])

        if save:
            self.save_xml(self._root_node, exp_type=exp_type, dataset=dataset)
        return 0

    # def del_xml_node(self, name: str):
    #
    #     exps = list(self._root_node.iterfind('experiment'))
    #     for node in exps:
    #         if node.attrib['name'] == name:
    #             self._root_node.remove(node)
    #         else:
    #             print(f'Node {name} does not exist.')
    #     self.save_xml()


class XMLProcessorLSB(XMLProcessor):
    def __init__(self, save_folder,
                 meta_params=('bl_config', 'bl_config_ref', 'dataset', 'n_sampl', 'nu', 'bl_type',
                              'bl_type_ref', 'target_', 'n_start', 'n_bags', 'comb_me', 'binary_mode',
                              'data'),
                 iteration_nodes=('mp_value', 'bl_accuracy_i', 'miss_train', 'miss_val', 'avg_div', 'alpha'),
                 mlp_bl_params=('inp_dim', 'nn_arch', 'le_rate', 'batch_s', 'n_episo', 'loss_fu', 'activa', 'metric_'),
                 cnn_bl_params=('archit', 'inp_dim', 'le_rate', 'loss_fu', 'batch_s', 'n_episo', 'metric_', 'n_filte',
                                's_filte', 'pooling_filter', 'activa', 'multila'),
                 umap_bl_params=('n_components', 'min_dist', 'local_connectivity', 'metric', 'target_metric',
                                 'n_neighbors', 'learning_rate'),
                 tree_bl_params=('max_dep', 'max_fea', 'max_lea', 'min_imp', 'c_weigh'),
                 deep_meta_nodes=('bl_config', 'bl_config_ref')):
        """
        - Saves and loads (configs) and (configs+data)
        - Changed to account for master model nodes
        :param save_folder: Folder in which XML object will be stored
        :param meta_params: Nodes to be created directly under root node
        :param iteration_nodes: Nodes to be created in each iteration node under phase node
        :param mlp_bl_params: Configurations for MLP BL
        :param cnn_bl_params: Configurations for CNN BL
        :param umap_bl_params: Configuration for UMAP BL
        :param tree_bl_params: Configurations for tree (CART) algorithm
        :param deep_meta_nodes: Stores in att. self.deep_nodes to know which node texts to skip when creating meta-nodes
        """
        super(XMLProcessorLSB, self).__init__(save_folder, meta_params, iteration_nodes, mlp_bl_params,
                                              cnn_bl_params, tree_bl_params, deep_meta_nodes)
        self._umap_bl_params = umap_bl_params

    def save_in_one_file(self, exp: dict, file_name):
        type_gen = exp['bl_type'].lower()
        type_ref = exp['bl_type_ref'].lower()
        type_master = exp['master_type'].lower()
        exp_type = 'LSB_' + type_gen + '_' + type_ref + '_' + type_master
        dataset = exp['dataset'].lower()
        self.save_xml(self._root_node, exp_type=exp_type, dataset=dataset, file_name=file_name)

    def write_all(self, exp: dict, res_data: tuple, xml_name='', save=False) -> int:
        """
        - Write the Experiment Configuration and Results in a XML File
        - Works by creating sub-nodes under nodes, emtpy sub-nodes and filling nodes with values
        :param exp: Nested dict Structure with Empty Node "data"
        :param res_data: Tuple made from return values of Class ResultLogger
        :param xml_name: Name of File
        :param save: Determines whether XML Node is saved when invoking the method
        :return: Error Indicator
        """
        iter_i = 0

        # Set up dict() for BL parameter choice
        param_map = {'mlp': self._mlp_bl_params, 'cnn': self._cnn_bl_params, 'tree': self._tree_bl_params,
                     'umap': self._umap_bl_params}

        # Configuration
        bl_type_gen = exp['bl_type'].lower()
        bl_type_ref = exp['bl_type_ref'].lower()
        master_type = exp['master_type'].lower()

        node_gen = param_map[bl_type_gen]
        node_ref = param_map[bl_type_ref]
        node_master = param_map[master_type]

        exp_i = ET.SubElement(self._root_node, 'experiment')
        exp_i.set('Variant', xml_name)
        node_objs_meta = []

        # Subnodes of 'configs'
        node_configs = []

        # Strings in File Name
        exp_type = bl_type_gen + '_' + bl_type_ref + '_' + master_type
        dataset = exp['dataset'].lower()

        # Write the meta-parameters, 'configs' must be at pos. -2
        for node in self._meta_params:
            node_obj_meta_i = self.create_subnode_with_skip(node=exp_i, sub_node_name=node,
                                                            sub_node_value=str(exp[node]),
                                                            skip_nodes=self.deep_nodes)
            node_objs_meta.append(node_obj_meta_i)

        # Create model config nodes
        for configs_subnode in exp['configs']:
            configs_i = self.create_subnode(node=node_objs_meta[-2], sub_node_name=configs_subnode,
                                            sub_node_value='')
            node_configs.append(configs_i)

        # Write gen. BL configuration as first sub-node under 'configs'
        for sub_node_gen in node_gen:
            # Triple Key
            sub_node_value = str(exp['configs']['bl_config'][sub_node_gen])
            self.create_subnode(node=node_configs[0], sub_node_name=sub_node_gen, sub_node_value=sub_node_value)

        # Write ref. BL configuration as second sub-node under 'configs'
        for sub_node_ref in node_ref:
            # Triple Key
            sub_node_value = str(exp['configs']['bl_config_ref'][sub_node_ref])
            self.create_subnode(node=node_configs[1], sub_node_name=sub_node_ref, sub_node_value=sub_node_value)

        # Write master model configuration as third sub-node under 'configs'
        for sub_node_master in node_master:
            # Triple Key
            if sub_node_master in ['loaded_', 'inp_dim']:
                sub_node_value = ''
            else:
                sub_node_value = str(exp['configs']['master_config'][sub_node_master])
            self.create_subnode(node=node_configs[2], sub_node_name=sub_node_master, sub_node_value=sub_node_value)

        # Chosen results, iter results parts should be in the beginning
        initial_acc_train, initial_acc_val, ref_acc_train, ref_acc_val, extractor_output_shape,  \
             ens_acc_train, ens_acc_val, gen_time, ref_time, perf_time, gen_end, n_miss = res_data

        # Concatenate initial and ref acc. for both train and val
        acc_train = initial_acc_train + ref_acc_train
        acc_val = initial_acc_val + ref_acc_val

        # Note: Order is important in iter_values
        # iter_values = [initial_acc_train, initial_acc_val, ref_acc_train, ref_acc_val, extractor_output_shape,
        #                ens_acc_train, ens_acc_val]
        iter_values = [acc_train, acc_val, extractor_output_shape, ens_acc_train, ens_acc_val]

        # Create node + value to store algo_time under 'data'
        self.create_subnode(node=node_objs_meta[-1], sub_node_name='algo_time',
                            sub_node_value=str(perf_time))

        # Create node + value  to store gen_time under 'data'
        self.create_subnode(node=node_objs_meta[-1], sub_node_name='gen_time',
                            sub_node_value=str(gen_time))

        # Create node + value to store ref_time under 'data'
        self.create_subnode(node=node_objs_meta[-1], sub_node_name='ref_time',
                            sub_node_value=str(ref_time))

        # Create gen_phase node
        gen_node = self.create_subnode(node=node_objs_meta[-1], sub_node_name='gen_phase', sub_node_value='')

        # Create ref_phase node
        ref_node = self.create_subnode(node=node_objs_meta[-1], sub_node_name='ref_phase', sub_node_value='')

        # Create the gen_phase sub-nodes (iter_n)
        gen_phase_iter_nodes = []
        for i in range(gen_end):
            iter_name = 'iter_' + str(iter_i)
            iter_node_i = self.create_subnode(gen_node, iter_name, '')
            gen_phase_iter_nodes.append(iter_node_i)
            iter_i += 1

        # Iterate over gen_phase iter nodes and add their values
        for idx, iter in enumerate(gen_phase_iter_nodes):
            for outer_inner, gen_iter_node in enumerate(self._iterations):
                self.create_subnode(iter, gen_iter_node, str(iter_values[outer_inner][idx]))

        # Edge case: iter_{ref} = 0, keep the structure
        # ref_iter_length = len(mp_val)
        ref_iter_length = len(ref_acc_train)
        if ref_iter_length == 0:
            ref_iter_length = 1
            iter_values = [' '] * len(iter_values)
        # / Edge Case

        # Create the ref_phase sub_nodes (iter_n)
        ref_phase_iter_objs = []
        for i in range(ref_iter_length):
            ref_iter_name = f'iter_{str(iter_i)}'
            ref_iter_node_i = self.create_subnode(ref_node, ref_iter_name, '')
            ref_phase_iter_objs.append(ref_iter_node_i)
            iter_i += 1

        # Iterate over refinement_iter_i and add values in sub3-nodes
        for idx, iter in enumerate(ref_phase_iter_objs):
            for outer_idx, ref_iter_node in enumerate(self._iterations):
                self.create_subnode(iter, ref_iter_node, str(iter_values[outer_idx][idx+gen_end]))

            # Proto
            self.create_subnode(node=iter, sub_node_name='n_miss', sub_node_value=str(n_miss[idx]))

        if save:
            self.save_xml(self._root_node, exp_type=exp_type, dataset=dataset)

        return 0

    def load_all(self, xml_path, n_exps=None):
        self._xml_path = xml_path
        root = ET.parse(self._xml_path)
        configs = []
        bl_config = []
        bl_config_ref = []
        master_config = []
        data = []
        rest = []

        d_1 = dict()
        d_config_2 = dict()
        d_ref_config_2 = dict()
        d_data_2 = dict()
        d_iter_gen_3 = dict()
        d_iter_ref_3 = dict()
        d_iter_val_4 = dict()

        for exp_idx, experiments in enumerate(root.iterfind('experiment')):
            for sub_node_gen in experiments.iterfind('bl_config'):
                for element in sub_node_gen:
                    string_val = element.text
                    try:
                        d_config_2[element.tag] = eval(string_val)
                    except:
                        d_config_2[element.tag] = string_val
                bl_config.append(d_config_2)
                d_config_2 = {}

            for sub_node_ref in experiments.iterfind('bl_config_ref'):
                for element in sub_node_ref:
                    string_val = element.text
                    try:
                        d_ref_config_2[element.tag] = eval(string_val)
                    except:
                        d_ref_config_2[element.tag] = string_val

                bl_config_ref.append(d_ref_config_2)
                d_ref_config_2 = {}

            # Load data to hierarchical dictionary, made of sub-dics
            data_length = len(experiments.find('data'))

            for sub_node_data_2 in experiments.iterfind('data'):
                for element in sub_node_data_2:
                    string_val = element.text
                    try:
                        d_data_2[element.tag] = eval(string_val)
                    except:
                        d_data_2[element.tag] = string_val

                # Find all iters in bagging/boosting phase and store in tmp lvl4 dictionary
                for sub3_iter_bagging in experiments.find('data').find('bagging_phase'):
                    # Lvl4: Values
                    for sub4_val_bagging in sub3_iter_bagging:
                        string_val = sub4_val_bagging.text
                        try:
                            d_iter_val_4[sub4_val_bagging.tag] = eval(string_val)
                        except:
                            d_iter_val_4[sub4_val_bagging.tag] = string_val
                    d_iter_gen_3[sub3_iter_bagging.tag] = d_iter_val_4
                    d_iter_val_4 = {}

                # Find all phases in one data node and store th lvl3 nodes in a dict
                for sub3_iter_boosting in experiments.find('data').find('boosting_phase'):
                    # Lvl4: Values
                    for sub4_val_boosting in sub3_iter_boosting:
                        string_val = sub4_val_boosting.text
                        try:
                            d_iter_val_4[sub4_val_boosting.tag] = eval(string_val)
                        except:
                            d_iter_val_4[sub4_val_boosting.tag] = string_val
                    d_iter_ref_3[sub3_iter_boosting.tag] = d_iter_val_4
                    d_iter_val_4 = {}

                d_data_2['bagging_phase'] = d_iter_gen_3
                d_data_2['boosting_phase'] = d_iter_ref_3

                data.append(d_data_2)
                d_data_2 = {}

            for sub_node_rest in experiments[2:]:
                sval = sub_node_rest.text
                try:
                    d_1[sub_node_rest.tag] = eval(sval)
                except:
                    d_1[sub_node_rest.tag] = sval

            # d_1['bl_config'] = bl_config[exp_idx]
            # d_1['bl_config_ref'] = bl_config_ref[exp_idx]
            # d_1['data'] = data[exp_idx]
            d_1['bl_config'], = bl_config
            d_1['bl_config_ref'], = bl_config_ref
            d_1['data'], = data

            rest.append(d_1)

            d_1 = {}
            bl_config = []
            bl_config_ref = []
            d_iter_gen_3 = {}
            d_iter_ref_3 = {}
            data = []

        # rest, = rest
        if n_exps:
            rest = rest[:n_exps]
        else:
            rest = rest[0]

        return rest



if __name__ == '__main__':
    pass


