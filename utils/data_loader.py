import numpy as np
import os


class DataLoader:
    def __init__(self, dataset_name: str, soda_main_dir, binary_labels: False, target=None, non_target=None, targets=None):
        """
        :param dataset_name: Name of dataset to be loaded
        :param binary_labels: If True, loads target and non-target only
        :param soda_main_dir: Dir in which subdir 'dataset' is located
        :param target: Target label
        :param non_target: Label to be discriminated against
        """
        self.dataset_name = dataset_name
        self.binary_labels = binary_labels
        self.target = target
        self.targets = targets
        self.non_target = non_target
        self.curr_dir = os.getcwd()
        # self.main_dir = os.path.dirname(self.curr_dir)
        self.main_dir = soda_main_dir
        self.binary_data_path = os.getcwd() + '/dataset/cifar10/binary_data.npy'

        if self.dataset_name not in ['mnist', 'cifar10']:
            raise NameError(f'Dataset {self.dataset_name,} not known!')

    def save_binary_data(self, binary_data: np.ndarray):
        np.save(file=self.binary_data_path, arr=binary_data)

    def get_binary_targets(self, x_r, y_r, x_r_val, y_r_val, t, nt, load_data=False):
        """
        - Get the raw features and labels for target t and non-target nt
        :param x_r: Raw features of train data
        :param y_r:  Labels of train data
        :param x_r_val: Raw features of validation data
        :param y_r_val: Labels of validation data
        :param t: One label in the binary task
        :param nt: Other label in the binary task
        :param load_data: If True, it will load a binary data set instead of creating one with another order of data
        :return: All the features and labels corresponding with target and non-target
        """
        if load_data:
            binary_dataset = np.load(self.binary_data_path, allow_pickle=True)
            x_r, y_r, x_r_val, y_r_val = binary_dataset
            return x_r, y_r, x_r_val, y_r_val

        target_idx, = np.where(y_r == t)
        non_target_idx, = np.where(y_r == nt)
        all_idx = np.concatenate((target_idx, non_target_idx), axis=0)
        np.random.shuffle(all_idx)
        y_r = y_r[all_idx]
        x_r = x_r[all_idx]
        # Validation data
        target_idx_val, = np.where(y_r_val == t)
        non_target_idx_val, = np.where(y_r_val == nt)
        all_idx_val = np.concatenate((target_idx_val, non_target_idx_val), axis=0)
        np.random.shuffle(all_idx_val)
        y_r_val = y_r_val[all_idx_val]
        x_r_val = x_r_val[all_idx_val]

        return x_r, y_r, x_r_val, y_r_val

    @staticmethod
    def get_specific_targets(x_raw, y_raw, x_raw_val, y_raw_val, target_labels: tuple):
        """
        - A more generalized version of self.get_binary_targets where an arbitrary number of labels can be chosen
        :param x_raw: Raw features of train data
        :param y_raw:  Labels of train data
        :param x_raw_val: Raw features of validation data
        :param y_raw_val: Labels of validation data
        :param target_labels: All labels to be included in the new set
        :return: All the features and labels corresponding with targets
        """
        targets_idx_head = np.array([])
        for target in target_labels:
            target_idx, = np.where(y_raw == target)
            targets_idx_head = np.concatenate((targets_idx_head, target_idx), axis=0)
        np.random.shuffle(targets_idx_head)
        y_raw = y_raw[targets_idx_head.astype(np.int)]
        x_raw = x_raw[targets_idx_head.astype(np.int)]

        targets_idx_head_val = np.array([])
        for target in target_labels:
            target_idx_val, = np.where(y_raw_val == target)
            targets_idx_head_val = np.concatenate((targets_idx_head_val, target_idx_val), axis=0)
        np.random.shuffle(targets_idx_head_val)
        y_raw_val = y_raw_val[targets_idx_head_val.astype(np.int)]
        x_raw_val = x_raw_val[targets_idx_head_val.astype(np.int)]

        return x_raw, y_raw, x_raw_val, y_raw_val

    def load_data(self, load_old_binary=False):
        if self.dataset_name == 'mnist':
            x_raw = np.load(self.main_dir + '/dataset/mnist/x.npy')
            y_raw = np.load(self.main_dir + '/dataset/mnist/y.npy')
            x_raw_val = np.load(self.main_dir + '/dataset/mnist/x_val.npy')
            y_raw_val = np.load(self.main_dir + '/dataset/mnist/y_val.npy')
            if self.binary_labels:
                x_raw, y_raw, x_raw_val, y_raw_val = self.get_binary_targets(x_raw, y_raw, x_raw_val, y_raw_val,
                                                                             self.target, self.non_target,
                                                                             load_data=load_old_binary)
            elif self.targets:
                x_raw, y_raw, x_raw_val, y_raw_val = self.get_specific_targets(x_raw, y_raw, x_raw_val, y_raw_val,
                                                                               self.targets)
        elif self.dataset_name == 'cifar10':
            x_raw = np.load(self.main_dir + '/dataset/cifar10/x_train_raw.npy')
            y_raw = np.load(self.main_dir + '/dataset/cifar10/y_train_raw.npy')
            x_raw_val = np.load(self.main_dir + '/dataset/cifar10/x_val_raw.npy')
            y_raw_val = np.load(self.main_dir + '/dataset/cifar10/y_val_raw.npy')
            if self.binary_labels:
                x_raw, y_raw, x_raw_val, y_raw_val = self.get_binary_targets(x_raw, y_raw, x_raw_val, y_raw_val,
                                                                             self.target, self.non_target,
                                                                             load_data=load_old_binary)
            elif self.targets:
                x_raw, y_raw, x_raw_val, y_raw_val = self.get_specific_targets(x_raw, y_raw, x_raw_val, y_raw_val,
                                                                               self.targets)

        else:
            raise NameError(f'Dataset {self.dataset_name,} not known!')

        return x_raw, y_raw, x_raw_val, y_raw_val


if __name__ == '__main__':
    # Run this script and set create_new_binary to True to generate new binary data set
    # TODO: Saving deliberate targets
    import os
    create_new_binary = False
    load_binary = True
    curr_dir = os.getcwd()
    soda_dir = os.path.dirname(curr_dir)
    dataset = 'cifar10'
    binary_mode = True
    target = 3
    non_target = 2
    data_loader = DataLoader(dataset_name=dataset, binary_labels=binary_mode, target=target,
                             non_target=non_target, soda_main_dir=soda_dir)

    x_raw, y_raw, x_raw_val, y_raw_val = data_loader.load_data(load_old_binary=load_binary)
    if create_new_binary:
        combined = np.asarray([x_raw, y_raw, x_raw_val, y_raw_val], dtype=object)
        data_loader.save_binary_data(combined)




