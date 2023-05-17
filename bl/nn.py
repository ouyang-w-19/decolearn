import os
import io
import keras.metrics
import tensorflow as tf
import numpy as np
from keras.layers import Dense, Input, Flatten, Dropout
from keras.models import load_model
from keras import backend as K
from bl.bl_base import BLBase
from typing import List, Union

import logging
logger = logging.getLogger('soda')

# TODO: Implement Batch-Normalization and Dropout-Learning


class NN(BLBase):

    def __init__(self, inp: tuple, layers: tuple, lr: float, loss: str, act: List[str],
                 dropout_rates: Union[List[float], None], metric: int, id: str, domain=None, baseline_pred=None):
        """
        - Instantiates a MLP object with given specs.
        :param inp: Dimensions for input layer
        :param layers: m nodes per n layer
        :param lr: Learning rate
        :param loss: Specifies how the loss is calculated
        :param act: Activation of each MLP layer
        :param metric: Specifies how model is evaluated in inference phase
        :param id: Identity of model
        :param domain: Deprecated
        :param baseline_pred: Deprecated
        """
        self._layers = layers
        self._act = act
        self._dropouts = dropout_rates
        self._metric = metric
        self._opti = None
        self._domain = domain
        self._domain_cache = None
        self._domain_activated = False
        self._baseline_pred = baseline_pred
        super(NN, self).__init__(inp, lr, loss, id)
        self.model_config = self._retrieve_model_info()
        self.input_layer_object = None
        self.layer_objects = None

    @classmethod
    def load_model(cls, path_name: str):
        return load_model(path_name)

    def switch_domain(self):
        if self._domain_activated:
            self._domain_cache = self._domain
            self._domain = None
            self._domain_activated = False
        else:
            self._domain = self._domain_cache
            self._domain_activated = True

    def save_model(self, path):
        self.model.save(path + '/' + self._id)

    def recompile_model(self):
        if self._metric == 0:
            metric = keras.metrics.BinaryAccuracy(name='Accuracy', threshold=0.5)
        elif self._metric == 1:
            metric = keras.metrics.SparseCategoricalAccuracy(name='Accuracy')
        else:
            # For testing only
            metric = keras.metrics.BinaryAccuracy(name='Accuracy', threshold=0.5)

        self.model.compile(optimizer=self._opti, loss=self._loss, metrics=metric)

    def build_model(self):
        """
        - Implements the abstract method
        :return: Unfitted MLP model with given specs.
        :rtype: Keras sequential model
        """
        idx = 1
        len_layers = len(self._layers)
        self._opti = tf.keras.optimizers.Adam(learning_rate=self._lr)
        model = keras.Sequential(name='NN_Nonlinear')

        model.add(Input(shape=self._input_dims, dtype=np.float64))
        model.add(Flatten())
        if self._dropouts:
            model.add(Dropout(self._dropouts[0], dtype=np.float64))

        # use tanh activation function for the last layer to generate output between -1 and 1
        for layer, act_i in zip(self._layers, self._act):
            last_layer = Dense(layer, activation=act_i, name=f'layer_{idx}', dtype=np.float64)
            model.add(last_layer)
            idx += 1
            if self._dropouts:
                if len_layers - idx >= 0:
                    model.add(Dropout(self._dropouts[idx-2], dtype=np.float64))

        if self._metric == 0:
            metric = keras.metrics.BinaryAccuracy(name='Accuracy', threshold=0.5)
        elif self._metric == 1:
            metric = keras.metrics.SparseCategoricalAccuracy(name='Accuracy')
        else:
            # For testing only
            metric = keras.metrics.BinaryAccuracy(name='Accuracy', threshold=0.5)

        model.compile(optimizer=self._opti, loss=self._loss, metrics=metric)

        return model

    def train(self, x, y, batch_size, epochs, validation=None, callbacks=None):
        """
        - Wrapper method to fit model of this class
        :param x: Feed forward data stored in np.ndarray
        :param y: Labels stored in np.ndarray
        :param batch_size: Mini batch size
        :param epochs: Number of iterations in which models are fitted on entire data set
        :param validation:
        :type validation:
        :param callbacks: Placeholder for callback objects (for logging, lr-schedule...etc.)
        """
        history = self.model.fit(x, y, batch_size=batch_size, epochs=epochs,
                                 callbacks=callbacks)
        self.binary_accuracy = history.history['Accuracy'][-1]

    def get_prediction(self, x, batch_size=None, binarize=False):
        """
        - Get prediction for input data, formatted to have at least three axes
        :param x: for forward pass
        :param batch_size: None: Max. speed, but also max. memory usage
        :param binarize: Binary mode: Switches to hard or soft predictions
        :return: Returns the models prediction
        """
        if x.ndim == 2:
            x = np.expand_dims(x, axis=0)
        if not binarize:
            preds = self.model.predict(x, batch_size)
        else:
            preds = np.array([1 if x > 0 else -1 for x in self.model.predict(x, batch_size)], dtype=np.short)
            preds = np.expand_dims(preds, axis=1)
        if self._domain is not None:
            self._domain_activated = True
            preds = preds * self._domain
        # Inverse outputs depending on the deviation from 0.5 in unfavorable direction
        # preds *= self.invert_factor
        return preds

    def get_r_prediction(self, x: Union[np.ndarray, List], batch_size=None):
        """Ignore validation dataset"""
        if None in self._domain or None in self._baseline_pred:
            raise Exception('Error: baseline_pred or domain not specified for method get_r_prediction()')
        if x.ndim == 2:
            x = np.expand_dims(x, axis=0)

        preds_on_m = self.model.predict(x, batch_size)

        column = np.copy(self._baseline_pred)
        column[self._domain, 0] = preds_on_m[self._domain, 0] - column[self._domain, 0]

        return column

    def _retrieve_model_info(self):
        stream = io.StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        summary_string = stream.getvalue()
        stream.close()
        # summary_string.split('SEP!')
        # text = ''
        # for line in summary_string.split('SEP!'):
        #     text = text + line + '\n'
        return summary_string


class NNFunc(NN):
    def __init__(self, inp: tuple, layers: tuple, lr: float, loss: str, act: List[str], dropout_rates,
                 metric: int, id: str, domain=None, baseline_pred=None):
        """
        - Inherits from class NN
        - Builds model with Keras Functional API instead of Sequential API
        :param inp: Dimensions for input layer
        :param layers: m nodes per n layer
        :param lr: Learning rate
        :param loss: Specifies how the loss is calculated
        :param act: Activation of each MLP layer
        :param metric: Specifies how model is evaluated in inference phase
        :param id: Identity of model
        :param domain: Deprecated
        :param baseline_pred: Deprecated
        """
        super(NNFunc, self).__init__(inp, layers, lr, loss, act, dropout_rates=dropout_rates, metric=metric,
                                     id=id, domain=domain, baseline_pred=baseline_pred)

    def build_model(self) -> keras.Model:
        """
        - Implements abstract method
        :return: Unfitted MLP model with given specs.
        :rtype: Keras functional model
        """
        len_layers = len(self._layers)
        self.input_layer_object = keras.Input(shape=self._input_dims, name=f'Input_0_{self._id}',
                                              dtype=np.float64)
        flatten = keras.layers.Flatten(name=f'Flatten_0_{self._id}')(self.input_layer_object)
        if self._dropouts:
            dropout_0 = keras.layers.Dropout(self._dropouts[0], dtype=np.float64)(flatten)
            layers_list = [self.input_layer_object, flatten, dropout_0]
        else:
            layers_list = [self.input_layer_object, flatten]

        mlp_id_layer = 1

        for mlp_layer, act_mlp_i in zip(self._layers, self._act):
            dense_i = keras.layers.Dense(mlp_layer, activation=act_mlp_i,
                                         name=f'MLP_{mlp_id_layer}_{self._id}',
                                         dtype=np.float64)(layers_list[-1])
            layers_list.append(dense_i)
            mlp_id_layer += 1
            if self._dropouts:
                if len_layers - mlp_id_layer >= 0:
                    dropout_i = keras.layers.Dropout(self._dropouts[mlp_id_layer-2], dtype=np.float64)(dense_i)
                    layers_list.append(dropout_i)

        self.layer_objects = layers_list
        model = keras.Model(inputs=self.input_layer_object, outputs=layers_list[-1])

        self._opti = tf.keras.optimizers.Adam(learning_rate=self._lr)

        if self._metric == 0:
            metric = keras.metrics.BinaryAccuracy(name='Accuracy', threshold=0.5)
        elif self._metric == 1:
            metric = keras.metrics.SparseCategoricalAccuracy(name='Accuracy')
        else:
            # For testing only
            metric = keras.metrics.BinaryAccuracy(name='Accuracy', threshold=0.5)

        model.compile(optimizer=self._opti, loss=self._loss, metrics=metric)

        return model

    def get_prediction(self, x: Union[np.ndarray, List[np.ndarray]], batch_size=None, binarize=False) -> np.ndarray:
        """
        - Enabled more general use than in base class
        - Format of x must be given correctly from script that invokes method
        - Removed domain-activated preds
        :param x: Data in ndarray-format or list for forward pass
        :param batch_size: None: Max. speed, but also max. memory usage
        :param binarize: Binary mode: Switches to hard or soft predictions
        :return: Returns the models prediction
        """
        if not binarize:
            preds = self.model.predict(x, batch_size)
        else:
            preds = np.array([1 if x > 0 else -1 for x in self.model.predict(x, batch_size)], dtype=np.short)
            preds = np.expand_dims(preds, axis=1)
        return preds


class NNModularInput(NNFunc):
    def __init__(self, inp: tuple, layers: tuple, lr: float, loss: str, act: List[str], dropout_rates,
                 metric: int, id: str, domain=None, baseline_pred=None):
        """
        - Implements NN model build with Add()-layer and Functional API
        - Used as base for special extensions as described in https://tinyurl.com/DeterministcMLSection312
        :param inp: Dimensions for input layer
        :param layers: m nodes per n layer
        :param lr: Learning rate
        :param loss: Specifies how the loss is calculated
        :param act: Activation of each MLP layer
        :param metric: Specifies how model is evaluated in inference phase
        :param id: Identity of model
        :param domain: Deprecated
        :param baseline_pred: Deprecated
        """
        super(NNModularInput, self).__init__(inp, layers, lr, loss, act, dropout_rates, metric, id, domain=domain,
                                             baseline_pred=baseline_pred)

    @staticmethod
    def dummy_activation(x):
        return tf.constant(0, dtype=np.float16) * x

    def build_model(self) -> keras.Model:
        """
        - Builds base of model with a modular input architecture
        - All lower parts must have ReLU activation
        - Base will be extended in each iteration
        :return: Modular input model base
        :rtype: Keras functional model
        """
        mlp_id_layer = 0
        len_layers = len(self._layers)
        self.input_layer_object = keras.Input(shape=self._input_dims, name=f'Input_0_{self._id}',
                                              dtype=np.float64)

        # ######### FOR TESTING INFLUENCE ONLY #########
        # test_input = keras.Input(shape=self._input_dims, name=f'Test_Input_{self._id}',
        #                          dtype=np.float64)
        # test_flatten = keras.layers.Flatten(name=f'Test_Flatten_{self._id}')(test_input)
        # dummy_layer = keras.layers.Dense(self._layers[0], activation=self.dummy_activation,
        #                                  name=f'Dummy_Layer',
        #                                  dtype=np.float16)(test_flatten)
        # ######### /FOR TESTING INFLUENCE ONLY #########

        flatten = keras.layers.Flatten(name=f'Flatten_0_{self._id}')(self.input_layer_object)
        if self._dropouts:
            dropout_0 = keras.layers.Dropout(self._dropouts[0], dtype=np.float64,
                                             name=f'Dropout_Sub_0_{self._id}')(flatten)
            layers_list = [self.input_layer_object, flatten, dropout_0]
        else:
            layers_list = [self.input_layer_object, flatten]

        dense_sub_layer = keras.layers.Dense(self._layers[0], activation=self._act[0],
                                             name=f'MLP_Sub_{mlp_id_layer}_{self._id}',
                                             dtype=np.float64)(layers_list[-1])
        # Dummy Layer
        dummy_layer = keras.layers.Dense(self._layers[0], activation=self.dummy_activation,
                                         name=f'Dummy_Layer',
                                         dtype=np.float16)(flatten)

        add_layer = keras.layers.Add(dtype=np.float64, name=f'Add_{self._id}')([dense_sub_layer, dummy_layer])

        #  layers_list = [self.input_layer_object, flatten, dense_sub_layer, dummy_layer, add_layer]
        layers_list.extend([dense_sub_layer, dummy_layer, add_layer])
        mlp_id_layer += 1

        for mlp_layer, act_mlp_i in zip(self._layers[1:], self._act[1:]):
            dense_i = keras.layers.Dense(mlp_layer, activation=act_mlp_i,
                                         name=f'MLP_{mlp_id_layer}_{self._id}',
                                         dtype=np.float64)(layers_list[-1])
            layers_list.append(dense_i)
            mlp_id_layer += 1
            if self._dropouts:
                if len_layers - mlp_id_layer-1 >= 0:
                    dropout_i = keras.layers.Dropout(self._dropouts[mlp_id_layer-2], dtype=np.float64,
                                                     name=f'Dropout_{mlp_id_layer}_{self._id}')(dense_i)
                    layers_list.append(dropout_i)

        self.layer_objects = layers_list
        model = keras.Model(inputs=[self.input_layer_object], outputs=layers_list[-1])

        self._opti = tf.keras.optimizers.Adam(learning_rate=self._lr)

        if self._metric == 0:
            metric = keras.metrics.BinaryAccuracy(name='Accuracy', threshold=0.5)
        elif self._metric == 1:
            metric = keras.metrics.SparseCategoricalAccuracy(name='Accuracy')
        else:
            # For testing only
            metric = keras.metrics.BinaryAccuracy(name='Accuracy', threshold=0.5)

        model.compile(optimizer=self._opti, loss=self._loss, metrics=metric)

        return model

    def train(self, x: Union[np.ndarray, List[np.ndarray]], y: Union[np.ndarray, List[np.ndarray]],
              batch_size, epochs, callbacks=None):
        """
        - Added additional typing in arguments lists
        :param x: Feed forward data stored in np.ndarray or list
        :param y: Labels stored in np.ndarray or list
        :param batch_size: Mini batch size
        :param epochs: Number of iterations in which models are fitted on entire data set
        :param callbacks: Placeholder for callback objects (for logging, lr-schedule...etc.)
        """
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs,
                       callbacks=callbacks)

    def recompile_model(self):
        if self._metric == 0:
            metric = keras.metrics.BinaryAccuracy(name='Accuracy', threshold=0.5)
        elif self._metric == 1:
            metric = keras.metrics.SparseCategoricalAccuracy(name='Accuracy')
        else:
            # For testing only
            metric = keras.metrics.BinaryAccuracy(name='Accuracy', threshold=0.5)

        self.model.compile(optimizer=self._opti, loss=self._loss, metrics=metric)


class NNRefine(NN):

    def __init__(self, inp, layers, lr, loss, act, metric, id: str, domain=None, baseline_pred=None):
        self._batch_u = None
        super(NNRefine, self).__init__(inp, layers, lr, loss, act, metric, id, domain=None, baseline_pred=None)
        self.model_config = self._retrieve_model_info()

    def linear_loss_core(self, y_true_comp, y_pred):
        # loss = y_true_comp * y_pred * (-1) * u,
        # but u is implemented as weight in training function
        return y_true_comp * y_pred * (-1)

    def train_with_linear_loss(self, x_train, y_train, batch_size, epochs, u,
                               callbacks=None):
        # implement u in loss function by taking u as sample weight
        # in model.fit() API
        # logger.info('----------Solve LPBoost pricing problem------------------')

        history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                 sample_weight=u, callbacks=callbacks)
        self.binary_accuracy = history.history['Binary Loss Accuracy'][-1]
        # logger.info('new bl specifications: todo')

    def build_model(self):
        idx = 1
        opti = tf.keras.optimizers.Adam(learning_rate=self._lr)
        model = keras.Sequential(name='NN_Linear_PP')

        model.add(Input(shape=self._input_dims))
        model.add(Flatten())

        # use tanh activation function for the last layer to generate output between -1 and 1
        for layer in self._layers[:-1]:
            # The
            last_layer = Dense(layer, activation=self._act, name='layer_' + str(idx),
                               kernel_initializer=keras.initializers.RandomNormal(mean=0, stddev=0.8))  # 0.5
            model.add(last_layer)
            idx += 1

        if self._metric == 0:
            model.add(Dense(self._layers[-1], activation='tanh', name='layer_out_tanh' + str(idx)))
            metric = keras.metrics.BinaryAccuracy(name='Accuracy', threshold=0)
        elif self._metric == 1:
            model.add(Dense(self._layers[-1], activation='sigmoid', name='layer_out_sigmoid_' + str(idx)))
            metric = keras.metrics.BinaryAccuracy(name='Accuracy', threshold=0.5)
        else:
            model.add(Dense(self._layers[-1], activation='softmax', name='layer_out_' + str(idx)))
            metric = keras.metrics.SparseCategoricalAccuracy(name='Accuracy')

        model.compile(optimizer=opti, loss=self.linear_loss_core, metrics=metric)

        return model


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time

    train_mode = True
    curr_dir = os.getcwd()
    main_dir = os.path.dirname(curr_dir)
    binary = True
    target_size = 6000
    stacks = int(60000/(2*target_size))

    x_train = np.load(main_dir + '/dataset/mnist/x.npy') / 255
    y_train = np.load(main_dir + '/dataset/mnist/y.npy')
    y_train_comp = np.where(y_train == 2, 1, -1)
    y_train_comp = np.expand_dims(y_train_comp, axis=1)
    y_train_comp = y_train_comp.astype('float64')

    x_val = np.load(main_dir + '/dataset/mnist/x_val.npy') / 255
    y_val = np.load(main_dir + '/dataset/mnist/y_val.npy')
    y_val_comp = np.where(y_val == 2, 1, -1)

    # target_idx, = np.where(y_train == 2)
    # complementary = np.asarray([idx for idx, val in enumerate(y_train) if val != 2])
    # idx_balanced = []
    # for i in range(stacks):
    #     comp = np.random.choice(complementary, target_size, replace=False)
    #     stack = np.concatenate((target_idx, comp))
    #     idx_balanced.append(stack)
    #
    # idx_balanced = np.asarray(idx_balanced)
    # idx_balanced = idx_balanced.reshape(idx_balanced.shape[0] * idx_balanced.shape[1])
    # np.random.shuffle(idx_balanced)

    if binary:
        y_train = np.where(y_train == 2, 1, 0)
        y_val_binary = np.where(y_val == 2, 1, 0)
        # y_train = np.where(y_train == 2, 1, -1)

    input_dims = (28, 28)
    learning_rate = 0.001
    if binary:
        metric = 0
        loss = 'binary_crossentropy'
        mlp = (4,1)
    else:
        metric = 1
        loss = 'sparse_categorical_crossentropy'
        mlp = (15, 10)
    epochs = 20

    # Train on subset and h(x_i) != 0 only if
    # domain = np.ones((len(y_train), 1))
    # domain[:100] = 0
    # nn = NN(input_dims, mlp, learning_rate, loss, metric, str(1))
    # nn.train(x_train[idx_balanced], y_train[idx_balanced], 50, 10, callbacks=None)
    # preds = nn.get_prediction(x_train, batch_size=None)
    # preds_val = nn.get_prediction(x_val)

    # # Test the residual idea
    # # y_train must be in {-1,1}
    # from copy import copy
    # tanh = keras.activations.tanh
    # y_train = np.expand_dims(y_train, axis=1)
    # y_train = y_train.astype('float32')
    # yh = preds * y_train
    # miss_idx, _ = np.where(yh <= 0)
    # len_data = x_train.shape[0]
    # complementary_miss_idx = [idx for idx in range(len_data) if idx not in miss_idx]
    # complementary_miss_idx = np.asarray(complementary_miss_idx)
    #
    # ri_raw = y_train[miss_idx] - preds[miss_idx]
    # ri = tanh(ri_raw).numpy()
    #
    # new_y = copy(y_train)
    # new_y[complementary_miss_idx] = -0.1
    # new_y[miss_idx] = ri
    #
    # nn_res = NN(input_dims, mlp, learning_rate, loss, metric, str(1))
    # nn_res.train(x_train, new_y, 50, 10, callbacks=None)
    # preds_res = nn_res.get_prediction(x_train)

    ### Test NNRefine ###
    u = np.ones((x_train.shape[0], 1))
    nn = NNRefine(input_dims, mlp, learning_rate, loss, metric, str(1))
    nn.train_with_linear_loss(x_train=x_train, y_train=y_train_comp, batch_size=100, epochs=epochs, u=u)
    preds = nn.get_prediction(x_train, batch_size=None)
    preds_val = nn.get_prediction(x_val)
    print(max(preds))

    y_val_comp = np.expand_dims(y_val_comp, axis=1)
    yh_val = preds_val * y_val_comp
    score_hist = np.where(yh_val <= 0, 1, 0)
    score = sum(score_hist)




