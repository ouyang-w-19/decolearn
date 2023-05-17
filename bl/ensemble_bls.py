import numpy as np
import keras
import io
import tensorflow as tf
from keras import layers
from tensorflow.keras.optimizers import Adam
from bl.cnn import CNNVGG
from typing import List, Union


class EnsembleCNNVGG(CNNVGG):
    def __init__(self, inp: tuple, lr, loss, metric, nfilter: tuple, sfilter: tuple, pooling_filter: tuple,
                 act: Union[str, List[str]], mlp: tuple,  id:str, x_train, y_train, batch_size, epochs, domain=None,
                 kernel_initializer=None):
        """
        - Configured for Binary classification only
        :param inp: Inp. dimension
        :param lr: Learning rate
        :param loss: Loss
        :param metric: Indicator for binary or mult
        :param nfilter: Number of filter for convolution ioeration per layer
        :param sfilter: Size of filter
        :param pooling_filter: Size of pooling filter
        :param act: Activation function
        :param mlp: MLP classifier nodes per layer
        :param id: Identifier of model
        :param x_train: Features of train data
        :param y_train: Labels of traind ata
        :param batch_size: Batch size
        :param epochs: n Epochs
        """
        # Train Data
        self.x_train = x_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.epochs = epochs

        self.n_bls = 3
        self._base_learners = list()
        self.currently_trained_bl = None

        super().__init__(inp, lr, loss, metric, nfilter, sfilter, pooling_filter,
                         act, mlp,  id, domain=domain, kernel_initializer=kernel_initializer)

        # Variance attributes
        self.var_calculated = False
        self.avg_var_ac = -1
        self.avg_var_nc = -1

        # BL Preds
        self.bl_preds = list()

        # Identifiers
        self.mlp_id = -1

    def calculate_diversity_term(self, avg_parameters, lambda_):
        """
        - Calculates diversity term according to || \bar{\theta} - \theta_i ||_2
        :param avg_parameters: Average parameters of the current ensemble
        :param lambda_: Coefficient of diversity term
        :return: Coefficient * diversity term
        """
        bli_weights = np.array([])
        for layer in self.currently_trained_bl.layers:
            w_i = layer.get_weights()
            if w_i:
                w_i_flat = np.concatenate((w_i[0].flatten(), w_i[1].flatten()), axis=0)
                bli_weights = np.concatenate((w_i_flat, bli_weights), axis=0)
        diff_vector = np.subtract(avg_parameters, bli_weights)
        diversity = np.linalg.norm(diff_vector, ord=2, axis=0)
        weighted_diversity = lambda_ * diversity

        return weighted_diversity

    def calculate_average_weights(self):
        """
        - Calculates average parameters of ensemble so far
        """
        all_weights = list()
        n_bls = self._base_learners.__len__()
        weights = np.array([])
        for bl in self._base_learners:
            for layer in bl.layers:
                w_i = layer.get_weights()
                # Concatenate weights from current and next weights
                if w_i:
                    w_i_flat = np.concatenate((w_i[0].flatten(), w_i[1].flatten()), axis=0)
                    weights = np.concatenate((w_i_flat, weights), axis=0)
            all_weights.append(weights)
            weights = list()
        all_weights = np.asarray(all_weights)
        average_weights = np.sum(all_weights, axis=0) / n_bls

        return average_weights

    def calculate_variance(self, preds, avg=True):

        if self._metric == 0:
            pred_var = preds.var(axis=0)
            if avg:
                pred_var = np.average(pred_var)
        else:
            max_pred = np.max(preds, axis=2)
            pred_var = max_pred.var(axis=0)
            pred_var = np.expand_dims(pred_var, axis=1)
            if avg:
                pred_var = np.average(pred_var)

        return pred_var

    def get_avg_variances(self, preds, miss_idx, non_miss_idx):
        # Variance of correctly classified
        all_correct = preds[:, non_miss_idx]
        avg_var_ac = self.calculate_variance(all_correct, avg=True)

        # Variance of incorrectly classified
        non_correct = preds[:, miss_idx]
        avg_var_nc = self.calculate_variance(non_correct, avg=True)

        return avg_var_ac, avg_var_nc

    def diverse_loss_generator(self):
        """
        - Wrapper function to generate new loss function
        :return: Updated loss function
        """
        avg_params = self.calculate_average_weights()
        diversity_term = self.calculate_diversity_term(avg_parameters=avg_params, lambda_=0.2)
        if self._metric == 0:
            def loss(y_true, y_pred):
                bce = keras.losses.BinaryCrossentropy()
                return bce(y_true, y_pred) - diversity_term
        else:
            def loss(y_true, y_pred):
                cce = keras.losses.CategoricalCrossentropy()
                return cce(y_true, y_pred) - diversity_term

        return loss

    def build_architecture(self):
        """
        - Builds the first BL which the others use as a starting point
        :return: Uncompiled BL
        """
        model = keras.Sequential(name='CNNVGG_Nonlinear_SP')
        ini = True
        conv_id = 1
        self.mlp_id = 1
        for stack_config, act_i in zip(self._n_filter, self._act):
            if ini:
                model.add(layers.Conv2D(stack_config, self._s_filter, padding='same', activation=act_i,
                                        input_shape=self._input_dims, name='VGG_Block_' + str(conv_id) + '_i',
                                        dtype=np.float64))
                model.add(layers.Conv2D(stack_config, self._s_filter, padding='same',
                                        activation=act_i, name='VGG_Block_' + str(conv_id) + '_ii',
                                        dtype=np.float64))
                model.add(layers.MaxPooling2D(pool_size=self._pooling_filter, dtype=np.float64,
                                              name=f'MaxPool_{conv_id}'))
                ini = False
                conv_id += 1
            else:
                model.add(layers.Conv2D(stack_config, self._s_filter, padding='same',
                                        activation=act_i, name='VGG_Block_' + str(conv_id) + '_i',
                                        dtype=np.float64))
                model.add(layers.Conv2D(stack_config, self._s_filter, padding='same',
                                        activation=act_i, name='VGG_Block_' + str(conv_id) + '_ii',
                                        dtype=np.float64))
                model.add(layers.MaxPooling2D(self._pooling_filter, dtype=np.float64, name=f'MaxPool_{conv_id}'))
                conv_id += 1
        model.add(layers.Flatten(name=f'Flatten_{conv_id}'))
        for mlp_layer, act_i in zip(self._mlp[:-1], self._act[conv_id:]):
            model.add(layers.Dense(mlp_layer, activation=act_i, name='layer_' + str(self.mlp_id), dtype=np.float64))
            self.mlp_id += 1

        return model

    def build_other_bls(self):
        """
        - Adds a copy of the first base learner to self._base_learners
        :return: Copy of first BL
        :rtype: keras.
        """
        new_bl = self.build_architecture()
        if self._metric == 0:
            # model.add(layers.Dense(self._mlp[-1], activation=self._act, name='layer_' + str(mlp_id)))
            new_bl.add(layers.Dense(self._mlp[-1], activation='sigmoid', name='layer_' + str(self.mlp_id),
                                      dtype=np.float64))
            # metric = keras.metrics.BinaryAccuracy(name='Accuracy', threshold=0)
            metric = keras.metrics.BinaryAccuracy(name='Accuracy', threshold=0.5)
            #  metric = keras.metrics.BinaryAccuracy(name='Binary Loss Accuracy')
        else:
            new_bl.add(layers.Dense(self._mlp[-1], activation='softmax', name='layer_' + str(self.mlp_id),
                                      dtype=np.float64))
            metric = keras.metrics.SparseCategoricalAccuracy(name='Accuracy')

        for layer in new_bl.layers:
            layer.set_weights(self._base_learners[0].get_layer(name=layer.name).get_weights())

        return new_bl

    def build_model(self):
        """
        - Invoked in __init__()
        - Also trains the BL
        - Training BL is necessary for building diverse BLs
        """
        first_bl = self.build_architecture()
        opt = Adam(learning_rate=self._lr)

        if self._metric == 0:
            # model.add(layers.Dense(self._mlp[-1], activation=self._act, name='layer_' + str(mlp_id)))
            first_bl.add(layers.Dense(self._mlp[-1], activation='sigmoid', name='layer_' + str(self.mlp_id),
                                      dtype=np.float64))
            # metric = keras.metrics.BinaryAccuracy(name='Accuracy', threshold=0)
            metric = keras.metrics.BinaryAccuracy(name='Accuracy', threshold=0.5)
            #  metric = keras.metrics.BinaryAccuracy(name='Binary Loss Accuracy')
        else:
            first_bl.add(layers.Dense(self._mlp[-1], activation='softmax', name='layer_' + str(self.mlp_id),
                                      dtype=np.float64))
            metric = keras.metrics.SparseCategoricalAccuracy(name='Accuracy')

        first_bl.compile(optimizer=opt, loss=self._loss, metrics=metric)
        first_bl.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs)
        self._base_learners.append(first_bl)

        n_datapoints = len(self.y_train)
        quarter = int(n_datapoints / 4)
        half = int(n_datapoints / 2)

        for i in range(self.n_bls-1):
            bl_i = self.build_other_bls()
            self.currently_trained_bl = bl_i
            diverse_loss = self.diverse_loss_generator()
            bl_i.compile(optimizer=opt, loss=diverse_loss, metrics=metric)
            sub_idx = np.random.choice(self.x_train.__len__(), size=self.x_train.__len__()//10, replace=False)
            sub_x = self.x_train[sub_idx]
            sub_y = self.y_train[sub_idx]
            bl_i.fit(sub_x, sub_y, batch_size=self.batch_size, epochs=self.epochs)
            self.currently_trained_bl = bl_i

            diverse_loss_i = self.diverse_loss_generator()
            bl_i.compile(optimizer=opt, loss=diverse_loss_i, metrics=metric)
            bl_i.fit(self.x_train[:quarter], self.y_train[:quarter], batch_size=self.batch_size, epochs=self.epochs)

            diverse_loss_ii = self.diverse_loss_generator()
            bl_i.compile(optimizer=opt, loss=diverse_loss_ii, metrics=metric)
            bl_i.fit(self.x_train[quarter:half], self.y_train[quarter:half], batch_size=self.batch_size,
                     epochs=self.epochs)

            diverse_loss_iii = self.diverse_loss_generator()
            bl_i.compile(optimizer=opt, loss=diverse_loss_iii, metrics=metric)
            bl_i.fit(self.x_train[half:(half+quarter)], self.y_train[half:(half+quarter)], batch_size=self.batch_size,
                     epochs=self.epochs)

            diverse_loss_iv = self.diverse_loss_generator()
            bl_i.compile(optimizer=opt, loss=diverse_loss_iv, metrics=metric)
            bl_i.fit(self.x_train[(half + quarter):], self.y_train[(half + quarter):],
                     batch_size=self.batch_size,
                     epochs=self.epochs)

            self._base_learners.append(bl_i)

        return self._base_learners

    def train(self, x, y, batch_size, epochs, callbacks=None):
        print(f'BL of class {self.__class__.__name__} are trained upon instantiation')

    def get_prediction(self, x, batch_size=None, binarize=None):

        # Get variances
        if not self.var_calculated:
            self.var_calculated = True
            preds = list()
            for bl in self._base_learners:
                pred = bl.predict(self.x_train, batch_size=None)
                preds.append(pred)
            preds = np.asarray(preds)

            ens_pred = np.sum(preds, axis=0) / self.n_bls
            non_miss_idx, = np.where(np.argmax(ens_pred))
            miss_idx, = np.where(np.argmax(ens_pred, axis=1) != self.y_train)

            self.avg_var_ac, self.avg_var_nc = self.get_avg_variances(preds, non_miss_idx=non_miss_idx,
                                                                      miss_idx=miss_idx)
        x_preds = list()
        for bl in self._base_learners:
            x_preds.append(bl.predict(x))
        x_preds = np.asarray(x_preds)
        var_datapoints = self.calculate_variance(x_preds, avg=False)

        # Build binary mask
        if self.avg_var_ac > self.avg_var_nc:
            # Valid preds
            # mask = np.where(var_datapoints > self.avg_var_ac, 1, 0)

            # Invalid points
            mask, _ = np.where(var_datapoints < self.avg_var_ac)
        else:
            # Valid preds
            # mask = np.where(var_datapoints < self.avg_var_ac, 1, 0)

            # Invalid points
            mask, _ = np.where(var_datapoints > self.avg_var_ac)
        # masked_x_preds = x_preds * mask
        x_preds[:, mask] = 0.5

        # final_pred = np.sum(masked_x_preds, axis=0) / self.n_bls
        final_pred = np.sum(x_preds, axis=0) / self.n_bls

        return final_pred

    def _retrieve_model_info(self):
        stream = io.StringIO()
        self._base_learners[0].summary(print_fn=lambda x: stream.write(x + '\n'))
        summary_string = stream.getvalue()
        stream.close()

        return summary_string



