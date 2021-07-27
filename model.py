
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import keras.backend as K
import distutils
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
#import cv2


train = pd.read_csv('https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.csv')
test = pd.read_csv('https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.csv')
columns = (['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
            'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
            'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
            'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack', 'level'])

train.columns = columns
test.columns = columns
train.loc[train.attack == 'normal', 'is_attacked'] = 0
train.loc[train.attack != 'normal', 'is_attacked'] = 1
test.loc[test.attack == 'normal', 'is_attacked'] = 0
test.loc[test.attack != 'normal', 'is_attacked'] = 1

# Категориальные признаки
test_without_cat = test.drop(columns=['protocol_type', 'service', 'flag', 'num_outbound_cmds'])
train_without_cat = train.drop(columns=['protocol_type', 'service', 'flag', 'num_outbound_cmds'])
X_train_without_cat = train_without_cat.iloc[: ,0:37]
y_train_without_cat  = train_without_cat.is_attacked.to_frame()
X_test_without_cat  = test_without_cat.iloc[: ,0:37]
y_test_without_cat  = test_without_cat.is_attacked.to_frame()

# convert class vectors to binary class matrices
def create_model_dropout(n_inputs):

    inputs = keras.Input(shape=(n_inputs,), name="input")
    batch = layers.BatchNormalization()(inputs)
    layer1 = layers.Dense(256, activation="sigmoid", name="dense1")(batch)
    drop = layers.Dropout(0.5)(layer1)
    layer11 = layers.Dense(56, activation="relu", name="dense3")(drop)
    drop = layers.Dropout(0.5)(layer11)
    layer2 = layers.Dense(256, activation="sigmoid", name="dense2")(drop)
    batch = layers.BatchNormalization()(layer2)
    outputs = layers.Dense(1, activation="sigmoid", name="out")(batch)

    return keras.Model(inputs=inputs, outputs=outputs)

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

model = create_model_dropout(37)
model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer="ADAM", metrics=['accuracy'])
history = model.fit(X_train_without_cat, y_train_without_cat, batch_size=100, epochs=10,
          verbose=1,
          validation_data=(X_test_without_cat, y_test_without_cat))


score = model.evaluate(X_test_without_cat, y_test_without_cat, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('model.h5')
#model_json = model.to_json()

#with open("model.json", "w") as json_file:
  #json_file.write(model_json)

#model.save_weights("model.h5")