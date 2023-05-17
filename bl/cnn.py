import os
import io
import warnings

import keras.metrics
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras import layers
from keras.models import load_model, Model
from bl.bl_base import BLBase
from tensorflow.keras.optimizers import Adam
from typing import List, Tuple, Union
from miscellaneous.custom_exceptions import FormatException


# TODO: Implement Batch-Normalization and Dropout-Learning

class CNN(BLBase):
    """ Applies cnn for learning a hypothesis

        Build:
            - inp:          Input dimensions
            - n_filter:     Extractor
            - mlp:          Classifier

        (k0, k1,...,kn) n = number of stacks, k = number of filters
        (l,m) Filter size, determines for max pooling and kernel filters
        activation of convolutional
        (o1, o2,...op) p = number of layers in MLP, o = node number
        If metric = 0, then

    """
    def __init__(self, inp: tuple, lr, loss, metric, nfilter: tuple, sfilter: tuple, pooling_filter: tuple,
                 act: str, mlp: tuple,  id:str, domain=None, baseline_pred=None, kernel_initializer=None):
        inp = self.correction(inp)
        self._n_filter = nfilter
        self._s_filter = sfilter
        self._pooling_filter = pooling_filter
        self._act = act
        self._mlp = mlp
        self._loss = loss
        self._metric = metric
        self._k_initializer = kernel_initializer
        self._domain = domain
        self._baseline_pred = baseline_pred
        super(CNN, self).__init__(inp, lr, loss, id)
        self.model_config = self._retrieve_model_info()

    def update_learning_rate(self, new_lr):
        K.set_value(self.model.optimizer.learning_rate, new_lr)

    def train_with_linear_loss(self, x_train, y_train, batch_size, epochs, u,
                               callbacks=None):
        # implement u in loss function by taking u as sample weight
        # in model.fit() API
        history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                 sample_weight=u, callbacks=callbacks)
        self.binary_accuracy = history.history['Accuracy'][-1]

    def linear_loss_core(self, y_true, y_pred):
        return y_true * y_pred * (-1)

    @classmethod
    def load_model(cls, path_name: str):
        return load_model(path_name)

    @staticmethod
    def correction(inp):
        if len(inp) == 2:
            inp = np.append(np.asarray(inp), 1)
        return inp

    def save_model(self, path):
        self.model.save(path + '/' + self._id)

    def build_model(self):
        model = keras.Sequential(name='CNNV_Nonlinear_SP')
        ini = True
        conv_id = 1
        mlp_id = 1
        for stack_config in self._n_filter:
            if ini:
                model.add(layers.Conv2D(stack_config, self._s_filter, padding='same',
                                        activation=self._act, input_shape=self._input_dims,
                                        name='Conv_' + str(conv_id), dtype=np.float64))
                model.add(layers.MaxPooling2D(pool_size=self._pooling_filter, dtype=np.float64))
                ini = False
                conv_id += 1
            else:
                model.add(layers.Conv2D(stack_config, self._s_filter, padding='same',
                                        activation=self._act, name='Conv_' + str(conv_id), dtype=np.float64))
                model.add(layers.MaxPooling2D(self._pooling_filter))
                conv_id += 1
        model.add(layers.Flatten())
        for mlp_layer in self._mlp[:-1]:
            model.add(layers.Dense(mlp_layer, activation=self._act, name='layer_' + str(mlp_id), dtype=np.float64))
            mlp_id += 1

        if self._metric == 0:
            # model.add(layers.Dense(self._mlp[-1], activation=self._act, name='layer_' + str(mlp_id)))
            model.add(layers.Dense(self._mlp[-1], activation='sigmoid', name='layer_' + str(mlp_id), dtype=np.float64))
            # metric = keras.metrics.BinaryAccuracy(name='Accuracy', threshold=0)
            metric = keras.metrics.BinaryAccuracy(name='Accuracy', threshold=0.5)
        else:
            model.add(layers.Dense(self._mlp[-1], activation='softmax', name='layer_' + str(mlp_id), dtype=np.float64))
            metric = keras.metrics.SparseCategoricalAccuracy(name='Accuracy')

        opt = Adam(learning_rate=self._lr)

        model.compile(optimizer=opt, loss=self._loss, metrics=metric)

        return model

    def train(self, x, y, batch_size, epochs, callbacks=None):
        if x.ndim < 4:
            x = np.expand_dims(x, axis=3)
        history = self.model.fit(x, y, batch_size=batch_size, epochs=epochs,
                                 callbacks=callbacks)
        self.accuracy = history.history['Accuracy'][-1]

    def get_prediction(self, x, batch_size=None, binarize=False):
        """Parameters:
                binarize:   Maps the output to {-1,1}; only available in binary mode"""
        # CNN Inout: (batch_size, width, height, n_channels)
        if x.ndim == 3:
            x = np.expand_dims(x, axis=3)
        if not binarize:
            preds = self.model.predict(x, batch_size)
        else:
            preds = np.array([1 if x > 0 else -1 for x in self.model.predict(x, batch_size)], dtype=np.short)
            preds = np.expand_dims(preds, axis=1)
        if self._domain is not None:
            preds = preds * self._domain
        # Inverse outputs depending on the deviation from 0.5 in unfavorable direction
        # preds *=  self.invert_factor
        return preds

    def get_r_prediction(self, x, batch_size=None):
        pass

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


"""
CNN with VGG Architecture 
"""


class CNNVGG(CNN):
    def __init__(self, inp: tuple, lr, loss, metric, nfilter: tuple, sfilter: tuple, pooling_filter: tuple,
                 act: Union[str, List[str]], mlp: tuple,  id:str, domain=None, kernel_initializer=None):
        self._batch_u = None
        super().__init__(inp, lr, loss, metric, nfilter, sfilter, pooling_filter, act, mlp,  id,
                         kernel_initializer=kernel_initializer)

    def build_model(self):
        model = keras.Sequential(name='CNNVGG_Nonlinear_SP')
        ini = True
        conv_id = 0
        mlp_id = 1
        for stack_config, act_i in zip(self._n_filter, self._act):
            if ini:
                model.add(layers.Conv2D(stack_config, self._s_filter, padding='same', activation=self._act,
                                        input_shape=self._input_dims, name='VGG_Block_' + str(conv_id) + '_i',
                                        dtype=np.float64))
                model.add(layers.Conv2D(stack_config, self._s_filter, padding='same',
                                        activation=self._act, name='VGG_Block_' + str(conv_id) + '_ii',
                                        dtype=np.float64))
                model.add(layers.MaxPooling2D(pool_size=self._pooling_filter, dtype=np.float64))
                ini = False
                conv_id += 1
            else:
                model.add(layers.Conv2D(stack_config, self._s_filter, padding='same',
                                        activation=self._act, name='VGG_Block_' + str(conv_id) + '_i',
                                        dtype=np.float64))
                model.add(layers.Conv2D(stack_config, self._s_filter, padding='same',
                                        activation=self._act, name='VGG_Block_' + str(conv_id) + '_ii',
                                        dtype=np.float64))
                model.add(layers.MaxPooling2D(self._pooling_filter, dtype=np.float64))
                conv_id += 1
        model.add(layers.Flatten())
        for mlp_layer in self._mlp[:-1]:
            model.add(layers.Dense(mlp_layer, activation=self._act, name='layer_' + str(mlp_id), dtype=np.float64))
            mlp_id += 1
        if self._metric == 0:
            # model.add(layers.Dense(self._mlp[-1], activation=self._act, name='layer_' + str(mlp_id)))
            model.add(layers.Dense(self._mlp[-1], activation='sigmoid', name='layer_' + str(mlp_id), dtype=np.float64))
            # metric = keras.metrics.BinaryAccuracy(name='Accuracy', threshold=0)
            metric = keras.metrics.BinaryAccuracy(name='Accuracy', threshold=0.5)
            #  metric = keras.metrics.BinaryAccuracy(name='Binary Loss Accuracy')
        else:
            model.add(layers.Dense(self._mlp[-1], activation='softmax', name='layer_' + str(mlp_id), dtype=np.float64))
            metric = keras.metrics.SparseCategoricalAccuracy(name='Accuracy')

        opt = Adam(learning_rate=self._lr)

        model.compile(optimizer=opt, loss=self._loss, metrics=metric)
        # focal_loss = tfa.losses.SigmoidFocalCrossEntropy()
        # model.compile(optimizer=opt, loss=focal_loss, metrics=metric)

        return model


class CNNVGGRelu(CNNVGG):
    def __init__(self, inp: tuple, lr, loss, metric, nfilter: tuple, sfilter: tuple, pooling_filter: tuple,
                 act: Union[str, List[str]], mlp: tuple,  id:str, domain=None, kernel_initializer=None):
        self._batch_u = None
        super().__init__(inp, lr, loss, metric, nfilter, sfilter, pooling_filter, act, mlp,  id,
                         kernel_initializer=kernel_initializer, domain=domain)

    def build_model(self):
        model = keras.Sequential(name='CNNVGG_Nonlinear_SP')
        ini = True
        conv_id = 0
        mlp_id = 1
        for stack_config, act_i in zip(self._n_filter, self._act):
            if ini:
                model.add(layers.Conv2D(stack_config, self._s_filter, padding='same', activation=self._act,
                                        input_shape=self._input_dims, name='VGG_Block_' + str(conv_id) + '_i',
                                        dtype=np.float64))
                model.add(layers.Conv2D(stack_config, self._s_filter, padding='same',
                                        activation=self._act, name='VGG_Block_' + str(conv_id) + '_ii',
                                        dtype=np.float64))
                model.add(layers.MaxPooling2D(pool_size=self._pooling_filter, dtype=np.float64))
                ini = False
                conv_id += 1
            else:
                model.add(layers.Conv2D(stack_config, self._s_filter, padding='same',
                                        activation=self._act, name='VGG_Block_' + str(conv_id) + '_i',
                                        dtype=np.float64))
                model.add(layers.Conv2D(stack_config, self._s_filter, padding='same',
                                        activation=self._act, name='VGG_Block_' + str(conv_id) + '_ii',
                                        dtype=np.float64))
                model.add(layers.MaxPooling2D(self._pooling_filter, dtype=np.float64))
                conv_id += 1
        model.add(layers.Flatten())
        for mlp_layer in self._mlp[:-1]:
            model.add(layers.Dense(mlp_layer, activation=self._act, name='layer_' + str(mlp_id), dtype=np.float64))
            mlp_id += 1
        if self._metric == 0:
            # model.add(layers.Dense(self._mlp[-1], activation=self._act, name='layer_' + str(mlp_id)))
            model.add(
                layers.Dense(self._mlp[-1], activation='relu', name='layer_' + str(mlp_id), dtype=np.float64))
            # metric = keras.metrics.BinaryAccuracy(name='Accuracy', threshold=0)
            metric = keras.metrics.BinaryAccuracy(name='Accuracy', threshold=0.5)
            #  metric = keras.metrics.BinaryAccuracy(name='Binary Loss Accuracy')
        else:
            model.add(
                layers.Dense(self._mlp[-1], activation='softmax', name='layer_' + str(mlp_id), dtype=np.float64))
            metric = keras.metrics.SparseCategoricalAccuracy(name='Accuracy')

        opt = Adam(learning_rate=self._lr)

        model.compile(optimizer=opt, loss=self._loss, metrics=metric)
        # focal_loss = tfa.losses.SigmoidFocalCrossEntropy()
        # model.compile(optimizer=opt, loss=focal_loss, metrics=metric)

        return model


class SplittedCNNVGG(CNNVGG):
    def __init__(self, inp: tuple, lr: float, loss, metric, nfilter: tuple, sfilter: tuple, pooling_filter: tuple,
                 class_act_per_layer: Union[list, tuple], extractor_act_per_layer: Union[list, tuple], mlp: tuple,
                 id:str, kernel_initializer=None):
        """
        - All model building methods are implemented with Keras functional API
        :param inp: Input dimension
        :param lr: Learning rate of model (can be changed)
        :param loss: Loss function of model
        :param metric: Similar to loss, but is used to evaluate the performance in inference phase,
        :param nfilter: Number of kernels per layer
        :param sfilter: Size of the kernels in each layer
        :param pooling_filter: Size of max_pooling filter
        :param class_act_per_layer: Activation spec for each layer of the classifier
        :param extractor_act_per_layer: Activation spec for each layer of the extractor
        :param mlp: Architecture of the MLP (m nodes per n layer)
        :param id: Model identification number
        """
        self._act_ext = extractor_act_per_layer
        self.extractor_input = None
        self.classifier_input = None
        self.extractor = None
        self.classifier = None
        self.extractor_layers = list()
        self.classifier_layers = list()
        super().__init__(inp, lr, loss, metric, nfilter, sfilter, pooling_filter, class_act_per_layer, mlp, id,
                         kernel_initializer=kernel_initializer)

    @classmethod
    def _retrieve_submodel_info(cls, submodel: keras.Model):
        """
        Pass the file handle in as a lambda function to make it callable
        Print function to use. Defaults to print. It will be called on each line of the summary.
        """
        stream = io.StringIO()
        submodel.summary(print_fn=lambda x: stream.write(x + '\n'))
        summary_string = stream.getvalue()

        return summary_string

    def diversity_regularized_loss(self, div_co):
        pass

    def build_extractor(self):
        # input = keras.Input(shape=self._input_dims)
        self.extractor_input = keras.Input(shape=self._input_dims, dtype=np.float64)
        layers_list = [self.extractor_input]
        stack_id = 1

        if self._k_initializer:
            mean, stddev = self._k_initializer
            initializer = tf.keras.initializers.RandomNormal(mean=mean, stddev=stddev)
        else:
            initializer = tf.keras.initializers.GlorotUniform()

        for stack_config, act_ext_i in zip(self._n_filter, self._act_ext):
            layer_i_a = keras.layers.Conv2D(stack_config, self._s_filter, padding='same', activation=act_ext_i,
                                            name=f'VGG_Block_{stack_id}_i', kernel_initializer=initializer,
                                            dtype=np.float64)(layers_list[-1])
            layer_i_b = keras.layers.Conv2D(stack_config, self._s_filter, padding='same', activation=act_ext_i,
                                            name=f'VGG_Block_{stack_id}_ii', kernel_initializer=initializer,
                                            dtype=np.float64)(layer_i_a)
            max_pool_i = keras.layers.AveragePooling2D(pool_size=self._pooling_filter, name=f'Max_Pool_{stack_id}',
                                                       dtype=np.float64)(layer_i_b)
            layers_list.extend([layer_i_a, layer_i_b, max_pool_i])
            stack_id += 1

        self.extractor_layers = layers_list
        extractor = Model(inputs=self.extractor_input, outputs=layers_list[-1])

        return extractor

    def build_classifier(self):
        extractor_output_shape = self.get_extractor_output_shape()

        self.classifier_input = keras.Input(shape=extractor_output_shape,
                                            name=f'classifier_input_m_{self._id}',
                                            dtype=np.float64)

        flatten = keras.layers.Flatten(name=f'classifier_flatten_m_{self._id}',
                                       dtype=np.float64)(self.classifier_input)
        mlp_id_layer_ = 1

        layers_list = [self.classifier_input, flatten]

        for mlp_layer, act_class_i in zip(self._mlp, self._act):
            dense_i = keras.layers.Dense(mlp_layer, activation=act_class_i,
                                         name=f'layer_m_{self._id}_mlp_{mlp_id_layer_}',
                                         dtype=np.float64)(layers_list[-1])
            layers_list.append(dense_i)
            mlp_id_layer_ += 1

        self.classifier_layers = layers_list
        classifier = keras.Model(inputs=self.classifier_input, outputs=layers_list[-1])

        return classifier

    def compile_classifier(self):
        if self._metric == 0:
            metric = keras.metrics.BinaryAccuracy(name='Accuracy', threshold=0.5)
        else:
            metric = keras.metrics.SparseCategoricalAccuracy(name='Accuracy')

        opt = Adam(learning_rate=self._lr)
        self.classifier.compile(optimizer=opt, loss=self._loss, metrics=metric)

    def build_model(self):
        self.extractor = self.build_extractor()
        self.classifier = self.build_classifier()
        complete_model_input = keras.Input(shape=self._input_dims, name=f'complete_model_input_{self._id}')
        model = keras.Model(inputs=complete_model_input,
                            outputs=self.classifier(self.extractor(complete_model_input)))

        if self._metric == 0:
            metric = keras.metrics.BinaryAccuracy(name='Accuracy', threshold=0.5)
        else:
            metric = keras.metrics.SparseCategoricalAccuracy(name='Accuracy')

        opt = Adam(learning_rate=self._lr)

        model.compile(optimizer=opt, loss=self._loss, metrics=metric)

        return model

    def update_complete_model(self, new_input_model: keras.Model):
        """
        - Entails redefining extractor sub-model with extended input
        :param new_input_model: New uncompiled model that will be compiled and assigned to self.model
        """
        self.model = keras.Model(inputs=new_input_model.inputs,
                                 outputs=new_input_model.layers[-1])

    def get_extractor_output_shape(self) -> tuple:
        """
        - Gets the output shape of the last layer of the extractor
        :return: Extractor output shape as a tuple
        """
        extractor_output_shape = (self.extractor.output.shape[1],
                                  self.extractor.output.shape[2],
                                  self.extractor.output.shape[3])
        return extractor_output_shape

    def get_prediction_classifier(self, transformation, batch_size=None, binarize=False):
        extractor_output_shape = self.get_extractor_output_shape()
        if transformation.shape[1:] != extractor_output_shape:
            raise FormatException(format=transformation.shape[1:], expected_format=extractor_output_shape)
        if not binarize:
            output = self.classifier.predict(transformation, batch_size)
        else:
            output = np.array([1 if x > 0 else -1 for x in self.classifier.predict(transformation,
                             batch_size)], dtype=np.short)
            output = np.expand_dims(output, axis=1)

        return output

    def get_transformation(self, data, batch_size=None, binarize=False):
        if data.shape[1:] != self._input_dims:
            raise FormatException(format=data.shape[1:], expected_format=self._input_dims)
        if not binarize:
            transformation = self.extractor.predict(data, batch_size)
        else:
            transformation = np.array([1 if x > 0 else -1 for x in self.extractor.predict(data,
                                        batch_size)], dtype=np.short)
            transformation = np.expand_dims(transformation, axis=1)

        return transformation

    def save_extractor(self, path):
        self.extractor.save(path + '/' + self._id)

    def save_classifier(self, path):
        self.classifier.save(path + '/' + self._id)


class SplittedCNNModularInput(SplittedCNNVGG):
    def __init__(self, inp: List[tuple], lr: float, loss, metric, nfilter: tuple, sfilter: tuple, pooling_filter: tuple,
                 class_act_per_layer: Union[list, tuple], extractor_act_per_layer: Union[list, tuple], mlp: tuple,
                 lower_nfilter: int, lower_sfilter: tuple, lower_extractor_act: str, lower_pool_type: str,
                 id:str, kernel_initializer=None):
        """
        - CNN with separate lower and upper part of the extractor layer
        - More flexible: For upper and lower, pooling-filter and size can be configured per layer
        :param inp: Input dimensions
        :param lr: Learning rate
        :param loss: Loss function
        :param metric: Indicates binary or multi-class classifications
        :param nfilter: Upper number of filters per layer
        :param sfilter: Upper size of filter per layer, ex.: ((3,3), (2,2), ...))
        :param pooling_filter: Upper type and size of pooling per layer, ex.: (('avg', (3,3)), ('max', (3,3)), ...)
        :param class_act_per_layer: Activation of MLP per layer
        :param extractor_act_per_layer: Upper Layer. Extractor activation per layer
        :param mlp: MLP layers
        :param lower_nfilter: Lower extractor filters for all lower layers
        :param lower_sfilter: Lower size of filter for all lower layers
        :param lower_extractor_act: Activation for all lower layers
        :param id: Identifier of model
        :param kernel_initializer:
        """
        self.extractor_inputs = list()
        self.lower_nfilter = lower_nfilter
        self.lower_sfilter = lower_sfilter
        self.lower_pool_type = lower_pool_type
        self.lower_act = lower_extractor_act
        self.upper_pooling = pooling_filter
        super(SplittedCNNModularInput, self).__init__(inp, lr, loss, metric, nfilter, sfilter, pooling_filter,
              class_act_per_layer, extractor_act_per_layer, mlp, id, kernel_initializer=kernel_initializer)


    @staticmethod
    def correction(inp):
        """
        - Redecleared as bypass method for this class
        :param inp: Input dimensions given as tuple or list
        :return: inp
        """
        return inp

    @staticmethod
    def get_pooling_layer(pooling_type: str, pool_size: tuple, name: str, last_layer):
        """
        -
        :param pooling_type: Avg. or Max. 2D
        :param pool_size: Size of pooling kernel
        :param name: Name of pooling layer
        :param last_layer: Layer that pooling layer is connected to
        :return: Keras model up until pool_layer
        """
        if pooling_type == 'max':
            pool_layer = keras.layers.MaxPooling2D(pool_size=pool_size,
                                                   name=name,
                                                   dtype=np.float64)(last_layer)
        elif pooling_type == 'avg':
            pool_layer = keras.layers.AvgPool2D(pool_size=pool_size,
                                                name=name,
                                                dtype=np.float64)(last_layer)
        else:
            raise TypeError(f'ERROR: Type "{pooling_type}" not supported')

        return pool_layer

    @staticmethod
    def get_biggest_factor(dims: list):
        """
        - Finds biggest factor if constantly integer dividing by 2
        :param dims:  Integers to be reduced
        :rtype: Results of //2 and common factor
        """
        factors = list()
        max_iter = 1000000
        for dim in dims:
            factor = dim
            factors_bl_i = list()
            factors_bl_i.append(factor)
            iter = 0
            while factor != 1:
                iter += 1
                factor = factor // 2
                factors_bl_i.append(factor)
                if iter > max_iter:
                    print('Error: Max. number of iterations exceeded!')
                    return -1
            factors.append(factors_bl_i)
        min_factor = min(factors)
        min_idx = factors.index(min_factor)
        rest_factors = factors[:]
        rest_factors.remove(min_factor)

        len_factors = factors.__len__()
        for idx, min_i in enumerate(min_factor):
            indices = list()
            common = list()
            for rest_i in rest_factors:
                for rest_idx, rest_int_i in enumerate(rest_i):
                    if rest_int_i == min_i:
                        common.append(rest_int_i)
                        indices.append(rest_idx)
            if common.__len__() == len_factors - 1:
                indices.insert(min_idx, idx)
                return factors, common[0], indices

        print('An error occured!')

    def build_extractor(self):
        """
        - Builds the extractor in a splitted manner
        - The lower parts are interpreted as independent models, so that their weights can be loaded
        - The linking part is a Conv2D-Layer
        - The upper part is the rest of the extractor
        """
        sub_model_outputs = list()
        layers = list()
        layer_id = 2
        layer_sub_id = 1

        # Weight initialization method
        if self._k_initializer:
            mean, stddev = self._k_initializer
            initializer = tf.keras.initializers.RandomNormal(mean=mean, stddev=stddev)
        else:
            initializer = tf.keras.initializers.GlorotUniform()

        if set(self._input_dims).__len__() != 1:
            # Get dims
            dims = list()
            for input_i in self._input_dims:
                dims.append(input_i[0])
            factors, common, pos_necessary_reduction = self.get_biggest_factor(dims)

            # Define Sub-Layers
            for input_i, n_reduction_i in zip(self._input_dims, pos_necessary_reduction):
                stack_id = 1
                sub_input_i = keras.Input(shape=input_i, dtype=np.float64, name=f'Sub_{layer_sub_id}_Input')
                self.extractor_inputs.append(sub_input_i)
                sub_layers = [sub_input_i]

                for i in range(n_reduction_i):
                    sub_layer_i = keras.layers.Conv2D(self.lower_nfilter, self.lower_sfilter, padding='same',
                                                      activation=self.lower_act,
                                                      name=f'Sub_{layer_sub_id}_Conv2D_{stack_id}',
                                                      kernel_initializer=initializer, dtype=np.float64)(sub_layers[-1])

                    sub_pool_i = self.get_pooling_layer(pooling_type=self.lower_pool_type, pool_size=(2, 2),
                                                        name=f'Sub_{layer_sub_id}_Pool_{stack_id}',
                                                        last_layer=sub_layer_i)
                    sub_layers.append(sub_pool_i)
                    stack_id += 1

                sub_model_outputs.append(sub_layers[-1])
                layer_sub_id += 1
        else:
            for input_i in self._input_dims:
                sub_input_i = keras.Input(shape=input_i, dtype=np.float64, name=f'Sub_{layer_sub_id}_Input')
                self.extractor_inputs.append(sub_input_i)
                sub_model_outputs.append(sub_input_i)
                layer_sub_id += 1

        # Define linking part
        if sub_model_outputs.__len__() == 1:
            concat = sub_model_outputs[0]
        else:
            concat = keras.layers.concatenate(sub_model_outputs)
        # linking_layer = keras.layers.Conv2D(self._n_filter[0], self._s_filter[0], padding='same',
        #                                     activation=self._act[0], name=f'Conv2D_{layer_id}',
        #                                     kernel_initializer=initializer, dtype=np.float64)(sub_model_outputs)
        linking_layer = keras.layers.Conv2D(self._n_filter[0], self._s_filter[0], padding='same',
                                            activation=self._act[0], name=f'Conv2D_{layer_id}',
                                            kernel_initializer=initializer, dtype=np.float64)(concat)
        pool_type_upper, pool_size_upper = self.upper_pooling[0]
        linking_pool = self.get_pooling_layer(pool_type_upper, pool_size_upper,
                                              name=f'Pool_{layer_id}',
                                              last_layer=linking_layer)
        layer_id += 1
        layers.extend([linking_layer, linking_pool])

        # Upper Part
        for nfilter_i, act_i, pooling_i, sfilter in zip(self._n_filter[1:], self._act[1:], self.upper_pooling[1:],
                                                        self._s_filter):
            layer_i = keras.layers.Conv2D(nfilter_i, sfilter, padding='same', activation=act_i,
                                          name=f'Conv2D_{layer_id}',
                                          kernel_initializer=initializer, dtype=np.float64)(layers[-1])
            pool_type, pool_size = pooling_i
            pool_i = self.get_pooling_layer(pool_type, pool_size, name=f'Pool_{layer_id}', last_layer=layer_i)

            layers.extend([layer_i, pool_i])

            layer_id += 1

        extractor = Model(inputs=self.extractor_inputs, outputs=layers[-1])

        return extractor

    def build_model(self):
        self.extractor = self.build_extractor()
        self.classifier = self.build_classifier()

        input_id = 1
        complete_model_inputs = list()
        for input_dim in self._input_dims:
            complete_model_input_i = keras.Input(shape=input_dim,
                                                 name=f'complete_model_input_{self._id}_{input_id}')
            complete_model_inputs.append(complete_model_input_i)
            input_id += 1

        model = keras.Model(inputs=complete_model_inputs,
                            outputs=self.classifier(self.extractor(complete_model_inputs)))

        if self._metric == 0:
            metric = keras.metrics.BinaryAccuracy(name='Accuracy', threshold=0.5)
        else:
            metric = keras.metrics.SparseCategoricalAccuracy(name='Accuracy')

        opt = Adam(learning_rate=self._lr)

        model.compile(optimizer=opt, loss=self._loss, metrics=metric)

        return model

    def train(self, x: Union[list, np.ndarray], y: np.ndarray, batch_size: int, epochs: int, callbacks=None):
        history = self.model.fit(x, y, batch_size=batch_size, epochs=epochs,
                                 callbacks=callbacks)
        self.accuracy = history.history['Accuracy'][-1]

    def get_transformation(self, data: List, batch_size=None, binarize=False):
        """
        - Input must be a list, that required re-implementation of inherited method
        - Iterates over the shapes
        :param data: Input Data
        :param batch_size: Batch size
        :param binarize: Deprecated
        :return:
        """
        for d, org_inp in zip(data, self._input_dims):
            if d.shape[1:] != org_inp:
                raise FormatException(format=d.shape[1:], expected_format=self._input_dims)

        transformation = self.extractor.predict(data, batch_size)

        return transformation

    def get_prediction(self, x, batch_size=None, binarize=False):
        """
        - x is
        :param x:
        :param batch_size:
        :param binarize:
        :return: Preds of the entire model
        """
        preds = self.model.predict(x, batch_size)

        return preds


class CNNLinearLoss(CNN):
    def __init__(self, inp: tuple, lr, loss, metric, nfilter: tuple, sfilter: tuple, pooling_filter: tuple,
                 act: str, mlp: tuple,  id:str, domain=None, baseline_pred=None):
        super().__init__(inp, lr, loss, metric, nfilter, sfilter, pooling_filter, act, mlp, id, domain=domain,
                         baseline_pred=baseline_pred)

    def build_model(self):
        model = keras.Sequential(name='CNN_Linear_PP')
        ini = True
        conv_id = 1
        mlp_id = 1
        for stack_config in self._n_filter:
            if ini:
                model.add(layers.Conv2D(stack_config, self._s_filter, padding='same',
                                        activation=self._act, input_shape=self._input_dims,
                                        name='Conv_' + str(conv_id)))
                model.add(layers.MaxPooling2D(pool_size=self._pooling_filter))
                ini = False
                conv_id += 1
            else:
                model.add(layers.Conv2D(stack_config, self._s_filter, padding='same',
                                        activation=self._act, name='Conv_' + str(conv_id)))
                model.add(layers.MaxPooling2D(self._pooling_filter))
                conv_id += 1
        model.add(layers.Flatten())
        for mlp_layer in self._mlp[:-1]:
            model.add(layers.Dense(mlp_layer, activation=self._act, name='layer_' + str(mlp_id)))
            mlp_id += 1

        if self._metric == 0:
            model.add(layers.Dense(self._mlp[-1], activation=self._act, name='layer_' + str(mlp_id)))
            metric = keras.metrics.BinaryAccuracy(name='Accuracy', threshold=0)
        else:
            model.add(layers.Dense(self._mlp[-1], activation='softmax', name='layer_' + str(mlp_id)))
            metric = keras.metrics.SparseCategoricalAccuracy(name='Accuracy')

        opt = Adam(learning_rate=self._lr)

        model.compile(optimizer=opt, loss=self.linear_loss_core, metrics=metric)

        return model


class CNNVGGLinearLoss(CNN):
    def __init__(self, inp: tuple, lr, loss, metric, nfilter: tuple, sfilter: tuple, pooling_filter: tuple,
                 act: str, mlp: tuple,  id:str, domain=None):
        self._batch_u = 0
        super().__init__(inp, lr, loss, metric, nfilter, sfilter, pooling_filter, act, mlp,  id)

    def build_model(self):
        model = keras.Sequential(name='CNNVGG_Linear_PP')
        ini = True
        conv_id = 0
        mlp_id = 1
        for stack_config in self._n_filter:
            if ini:
                model.add(layers.Conv2D(stack_config, self._s_filter, padding='same', activation=self._act,
                                        input_shape=self._input_dims, name='VGG_Block_' + str(conv_id) + '_i'))
                model.add(layers.Conv2D(stack_config, self._s_filter, padding='same',
                                        activation=self._act, name='VGG_Block_' + str(conv_id) + '_ii'))
                model.add(layers.MaxPooling2D(pool_size=self._pooling_filter))
                ini = False
                conv_id += 1
            else:
                model.add(layers.Conv2D(stack_config, self._s_filter, padding='same',
                                        activation=self._act, name='VGG_Block_' + str(conv_id) + '_i'))
                model.add(layers.Conv2D(stack_config, self._s_filter, padding='same',
                                        activation=self._act, name='VGG_Block_' + str(conv_id) + '_ii'))
                model.add(layers.MaxPooling2D(self._pooling_filter))
                conv_id += 1
        model.add(layers.Flatten())
        for mlp_layer in self._mlp[:-1]:
            model.add(layers.Dense(mlp_layer, activation=self._act, name='layer_' + str(mlp_id)))
            mlp_id += 1
        if self._metric == 0:
            model.add(layers.Dense(self._mlp[-1], activation=self._act, name='layer_' + str(mlp_id)))
            metric = keras.metrics.BinaryAccuracy(name='Accuracy', threshold=0)
        else:
            model.add(layers.Dense(self._mlp[-1], activation='softmax', name='layer_' + str(mlp_id)))
            metric = keras.metrics.SparseCategoricalAccuracy(name='Accuracy')

        opt = Adam(learning_rate=self._lr)

        # model.compile(optimizer=opt, loss=self._loss, metrics=metric)
        # model.compile(optimizer=opt, loss=self.linear_loss_core, metrics=metric)
        model.compile(optimizer=opt, loss=self.linear_loss_core, metrics=metric)

        return model
