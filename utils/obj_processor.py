import _pickle as pickle
import numpy as np

# Saving, loading verified!


def unpickle_obj(file):
    with open(file, 'rb') as raw_bin:
        obj = pickle.load(raw_bin, encoding='bytes')
    return obj


def pickle_obj(obj):
    pass


def cifar_loader():
    n_train = 5
    all_train = []
    path_to_batches = 'C:/Users/vanya/OneDrive/Desktop/Temp/cifar-10-batches-py/'
    for idx in range(n_train):
        idx += 1
        file_name_train = path_to_batches + 'data_batch' + f'_{idx}'
        all_train.append(unpickle_obj(file_name_train))

    file_name_val = path_to_batches + 'test_batch'
    one_val = unpickle_obj(file_name_val)
    return all_train, one_val


def cifar_saver():
    train_data_l = []
    train_labels_l = []

    all_train, one_val = cifar_loader()

    for train_i in all_train:
        train_data_l.append(train_i[b'data'])
        train_labels_l.append(train_i[b'labels'])

    train_data_np = np.asarray(train_data_l)
    train_data_np = np.concatenate(train_data_np)
    train_data_np = np.reshape(train_data_np, (50000, 32, 32, 3))
    train_labels_np = np.asarray(train_labels_l)
    train_labels_np = np.concatenate(train_labels_np)

    val_data = one_val[b'data']
    val_data_np = np.asarray(val_data)
    val_data_np = np.reshape(val_data_np, (10000, 32, 32, 3))
    val_labels = one_val[b'labels']
    val_labels_np = np.asarray(val_labels)

    np.save('/soda/dataset/cifar10/x_train_raw.npy',
            train_data_np)
    np.save('/soda/dataset/cifar10/y_train_raw.npy',
            train_labels_np)

    np.save('/soda/dataset/cifar10/x_val_raw.npy',
            val_data_np)
    np.save('/soda/dataset/cifar10/y_val_raw.npy',
            val_labels_np)


# Testing
if __name__ == '__main__':
    cifar_loader()
    cifar_saver()

    x_train_raw = np.load('/soda/dataset/cifar10/x_train_raw.npy')
    y_train_raw = np.load('/soda/dataset/cifar10/y_train_raw.npy')

    x_train_raw = np.reshape(x_train_raw, (50000, 3, 32, 32))
    import matplotlib.pyplot as plt
    for i in range(10, 20, 1):
        plt.title(y_train_raw[i])
        plt.imshow(x_train_raw[i][0], cmap='gray')
        plt.show()
