import keras
import numpy as np
import tensorflow as tf
from keras.metrics import Accuracy


class MaskingLayer(keras.layers.Layer):
    def __init__(self, units=10, input_dim=10):
        super(MaskingLayer, self).__init__()
        w_init = np.zeros((units, input_dim))
        self.w = tf.Variable(shape=(units, input_dim), initial_value=w_init, trainable=False)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(units,), dtype='float32'), trainable=False)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) # + self.b


def build_linear_comb_with_add(bl_output_shape: tuple, n_bls: int, lr):
    bl_idx = 1
    inputs = list()
    flatten = list()

    for i in range(n_bls):
        input_layer_i = keras.Input(shape=bl_output_shape, name=f'BLInput_{bl_idx}', dtype=np.float64)
        flatten_i = keras.layers.Flatten(name=f'Flatten_Vector_{bl_idx}', dtype=np.float64)(input_layer_i)

        inputs.append(input_layer_i)
        flatten.append(flatten_i)

        bl_idx += 1

    add_layer = keras.layers.Add(dtype=np.float64, name='Add_Layer')(flatten)
    # softmax = keras.layers.Dense(bl_output_shape[0], activation='softmax', dtype=np.float64,
    #                              name='Softmax_Layer')(add_layer)


    model = keras.Model(inputs=inputs, outputs=add_layer)
    # model = keras.Model(inputs=inputs, outputs=softmax)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    metric = keras.metrics.SparseCategoricalAccuracy(name='Accuracy')

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=metric)

    return model


if __name__ == '__main__':
    # # For testing only
    # pred_vector = np.random.dirichlet(np.ones(10), size=10)
    # labels = np.array([1, 3, 5, 3, 8, 9, 0, 3, 5, 8])
    #
    # lcm = build_linear_comb_with_add(bl_output_shape=(10,), n_bls=10, lr=0.001)
    #
    # lcm.predict([pred_vector])

    x = tf.ones((10, 1))
    masking_layer = MaskingLayer(10, 1)
    y = masking_layer(x)
    print(y)

