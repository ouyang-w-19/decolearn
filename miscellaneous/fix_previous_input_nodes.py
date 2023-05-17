""""
Script to Test the Fix-Input-Nodes-Mode as described in the following document:
https://www.notion.so/Concept-Deterministic-ML-e8a3789abcd24e4d81163f2a2d88158a
"""
import keras.metrics
import os
import numpy as np
from bl.cnn import SplittedCNNVGG
from utils.data_loader import DataLoader
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Concatenate
from keras import layers
from keras.models import load_model, Model
from keras.metrics import Accuracy

# Get directories
curr_dir = os.getcwd()
main_dir = os.path.dirname(curr_dir)

# Set modes and Dataset
binary_mode = False
dataset = 'cifar10'
target = 3
non_target = 2

# Set configs
n_samples = 60000
input_dims = (32, 32, 3)
lr = 0.001
if binary_mode:
    loss = 'binary_crossentropy'
    metric = 0
    act = 'tanh'
    mlp = (3, 2, 1)
else:
    loss = 'sparse_categorical_crossentropy'
    metric = 1
    act = 'tanh'
    #mlp = (5, 10)
    mlp = (20, 10)

nfilter = (32, 16)
sfilter = (2, 2)
# Pooling: Size must be adapted to number of cnn layers
pooling_filter = (2, 2)
extractor_act = ['relu'] * len(nfilter)
class_act = ['relu'] * len(mlp)
class_act[-1] = 'softmax'

model_id_1 = 'split_1'
model_id_2 = 'split_2'
batch_size_limit = None
batch_size = 64
epochs = 1

# Load Data
data_loader = DataLoader(dataset_name=dataset, binary_labels=binary_mode, target=target,
                         non_target=non_target, soda_main_dir=main_dir)
x_train_raw, y_train, x_val_raw, y_val = data_loader.load_data()
x_train = x_train_raw / 255
x_val = x_val_raw / 255

""" 
Test SplittedCNNVGG 
"""
cnnvgg_s = SplittedCNNVGG(input_dims, lr, loss, metric, nfilter,
                          sfilter, pooling_filter, class_act, extractor_act, mlp, model_id_1)
cnnvgg_s.train(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=None)


ext_train = cnnvgg_s.get_transformation(x_train)
class_out = cnnvgg_s.get_prediction_classifier(ext_train)

# Instantiate Accuracy Object
acc = keras.metrics.Accuracy()

# # Complete Model Score
com_predicted_classes = np.argmax(class_out, axis=1)
acc.update_state(y_train, com_predicted_classes)
com_score = acc.result().numpy()
print(f'CNNVGG_si model train score is: {com_score}\n')


"""
Add output of an extra model to the complete model (Ensemble)
"""
opt = Adam(learning_rate=lr)
cnnvgg_sii = SplittedCNNVGG(input_dims, lr, loss, metric, nfilter,
                            sfilter, pooling_filter, class_act, extractor_act, mlp, model_id_2)
cnnvgg_sii.train(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=None)
new_ext_train = cnnvgg_sii.get_transformation(x_train)
new_class_out = cnnvgg_sii.get_prediction_classifier(new_ext_train)
acc.reset_states()
acc.update_state(y_train, np.argmax(new_class_out, axis=1))
new_class_score = acc.result().numpy()
print(f'CNNVGG_sii model train score is: {new_class_score}\n')

# # Concatenate the Predictions of BL models
# ext_con = np.concatenate([ext_train, new_ext_train], axis=-1)
#
# Manually generate \mathcal{G}
ext_train_flat = np.reshape(ext_train, (ext_train.shape[0],
                                        ext_train.shape[1] *
                                        ext_train.shape[2] *
                                        ext_train.shape[3]))
new_ext_train_flat = np.reshape(new_ext_train, (new_ext_train.shape[0],
                                                new_ext_train.shape[1] *
                                                new_ext_train.shape[2] *
                                                new_ext_train.shape[3]))
combined_input = np.concatenate((new_ext_train_flat, ext_train_flat), axis=1)


"""
Add Function to Change Model Architecture
"""


def change_model(model, new_shape):
    # rebuild model architecture by exporting and importing via json
    classifier_config = model.to_json()
    start_idx = classifier_config.find('batch_input_shape')
    end_idx = classifier_config.find(']', start_idx) + 1
    target_string = classifier_config[start_idx:end_idx]
    payload_string = 'batch_input_shape": [null, '
    for dim in new_shape[:-1]:
        payload_string += f'{dim}, '
    payload_string += f'{new_shape[-1]}]'
    new_config = classifier_config.replace(target_string, payload_string)
    new_model = keras.models.model_from_json(new_config)
    new_model.summary()
    # copy weights from old model to new one
    for layer in new_model.layers:
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
            print('Transfer complete')
        except:
            print("Could not transfer weights for layer {}".format(layer.name))

    return new_model


# classifier = cnnvgg_s.classifier
# extended_classifier = change_model(classifier, (16, 16, 128))

"""
Test Add()-Layer of Keras
"""
# Artificial Data
arr1 = np.array([[[1]*16]])
arr2 = np.array([[[2]*16]])

# Create Input and Add Them
inp1 = keras.layers.Input(shape=(1, 1, 16))
inp2 = keras.layers.Input(shape=(1, 1, 16))
add_layer = keras.layers.Add()([inp1, inp2])

# Build Model
model_add_test = keras.models.Model(inputs=[inp1, inp2], outputs=add_layer)

# Predict Output
model_add_test_pred = model_add_test.predict([arr1, arr2])
print(f'Output of Add-Layer is: {model_add_test_pred}')

"""
Test Fixed-Input-Node Idea
"""
# # Test Concatenated Model (Only if Wished to Calculate Losses Together)
# cnnvgg_s.model.trainable = False
# con_model_inp1 = cnnvgg_s.model.output
# con_model_inp2 = cnnvgg_sii.model.output
#
# con = Concatenate()([con_model_inp1, con_model_inp2])
# con_softmax = keras.layers.Dense(10, activation='softmax', name=f'con_softmax',
#                                  dtype=np.float64)(con)
# con_model = Model(inputs=[cnnvgg_s.model.input, cnnvgg_sii.model.input], outputs=[con_softmax])
# con_model.compile(optimizer=opt,
#                   loss={
#                       'con_softmax': loss
#                   },
#                   metrics=['accuracy'])
# con_model.fit({'complete_model_input_split_1': x_train,
#                'complete_model_input_split_2': x_train},
#               {'con_softmax': y_train})
# con_model_pred = con_model.predict([x_train, x_train])

# Test Model Connections to Lower Layers and then Add()
model_add_input1 = keras.Input(shape=(8, 8, 16), name='Add_Input_1')
model_add_input2 = keras.Input(shape=(8, 8, 16), name='Add_Input_2')

input1_flatten = keras.layers.Flatten(name=f'model_add_input1_flatten', dtype=np.float64)(model_add_input1)
input2_flatten = keras.layers.Flatten(name=f'model_add_input2_flatten', dtype=np.float64)(model_add_input2)

lower_ens_i = keras.layers.Dense(mlp[0], activation=class_act[0],
                                 name='lower_ens_i',
                                 dtype=np.float64)(input1_flatten)

lower_ens_ii = keras.layers.Dense(mlp[0], activation=class_act[0],
                                  name='lower_ens_ii',
                                  dtype=np.float64)(input2_flatten)

add_layer = keras.layers.Add(dtype=np.float64)([lower_ens_i, lower_ens_ii])

# Usually the upper part of the master model would start here

softmax = keras.layers.Dense(mlp[1], activation='softmax', name=f'add_softmax',
                             dtype=np.float64)(add_layer)

model_add = keras.models.Model(inputs=[model_add_input1, model_add_input2], outputs=softmax)
model_add.compile(optimizer=opt,
                  loss={
                      'add_softmax': loss
                  },
                  metrics=['accuracy'])

# Freeze lower master model layer associated with first input
model_add.layers[4].trainable = False

model_add.fit({'model_add_input_split_1': ext_train,
               'model_add_input_split_2': new_ext_train},
              {'add_softmax': y_train},
              batch_size=None)
model_add_pred = model_add.predict([ext_train, new_ext_train])
acc.reset_states()
acc.update_state(y_train, np.argmax(model_add_pred, axis=1))
print(f'add_model train accuracy is: {acc.result().numpy()}')




