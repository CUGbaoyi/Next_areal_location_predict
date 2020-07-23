#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import time

import numpy as np
import pandas as pd
from keras import metrics, callbacks
from keras.layers import Input, LSTM, Dense, concatenate, Bidirectional, Conv1D, GlobalMaxPooling1D, \
    GlobalAveragePooling1D, Dropout
from keras.models import Model
from keras.utils import np_utils


def pre_df(_d):
    """
    pre train d, transform to  array, (49, 205)
    :param _d:
    :return: (uid_id, lstm x, lstm y, valid y, aux data(time information which are not used in this task))
    """
    for uid_index, v in _d.items():
        v = np.array(v)
        # split the data to X and Y
        pre_X, y = v[:-2, 44:], category_dict[v[-2, 0]]

        # flatten a
        p = pre_X.flatten()
        # get nodata count, ready to padding
        padding_weight = (49 * 200 - p.shape[0])
        # padding and reshape to 49 * 200
        X = np.pad(p, (0, padding_weight), 'constant', constant_values=-1.).reshape(49, 200)

        return uid_index, X, y


def lstm_cnn(model_main_data, model_label):
    """
    build model
    :return:
    """
    print("Building lstm cnn model *** ")

    main_input = Input(shape=(49, 200), dtype="float32", name="main_input")

    # define lstm output size == 256
    lstm_out = Bidirectional(LSTM(256, return_sequences=True))(main_input)
    lstm_out = Dropout(0.5)(lstm_out)
    conv1d_2 = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(lstm_out)
    conv1d_3 = Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="he_uniform")(lstm_out)
    conv1d_5 = Conv1D(64, kernel_size=5, padding="valid", kernel_initializer="he_uniform")(lstm_out)
    avg_pool_2 = GlobalAveragePooling1D()(conv1d_2)
    max_pool_2 = GlobalMaxPooling1D()(conv1d_2)
    avg_pool_3 = GlobalAveragePooling1D()(conv1d_3)
    max_pool_3 = GlobalMaxPooling1D()(conv1d_3)
    avg_pool_5 = GlobalAveragePooling1D()(conv1d_5)
    max_pool_5 = GlobalMaxPooling1D()(conv1d_5)
    conc = concatenate([avg_pool_2, max_pool_2, avg_pool_3, max_pool_3, avg_pool_5, max_pool_5])
    conc = Dropout(0.5)(conc)
    main_output = Dense(category_len, activation='softmax', name='aux_output')(conc)

    model = Model(inputs=main_input, outputs=main_output)
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy",
                  metrics=[metrics.top_k_categorical_accuracy, 'accuracy'])
    # start to train
    print("start *** ")
    start_time = time.time()
    tensorboard = callbacks.TensorBoard(log_dir=f'logs_{category_len}')
    callback_lists = [tensorboard]
    history = model.fit(model_main_data, model_label, epochs=40, batch_size=128,
                        validation_split=0.1, callbacks=callback_lists)
    print(max(history.history['val_acc']))
    print(max(history.history['val_top_k_categorical_accuracy']))
    print("save model")
    spend_time = time.time() - start_time
    print(spend_time)
    model.save("./data/lstm_cnn.h5")


def only_cnn(model_main_data, model_label):
    print("Building only cnn model *** ")

    main_input = Input(shape=(49, 200), dtype="float32", name="main_input")

    # define lstm output size == 256
    conv1d_2 = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(main_input)
    conv1d_3 = Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="he_uniform")(main_input)
    conv1d_5 = Conv1D(64, kernel_size=5, padding="valid", kernel_initializer="he_uniform")(main_input)
    avg_pool_2 = GlobalAveragePooling1D()(conv1d_2)
    max_pool_2 = GlobalMaxPooling1D()(conv1d_2)
    avg_pool_3 = GlobalAveragePooling1D()(conv1d_3)
    max_pool_3 = GlobalMaxPooling1D()(conv1d_3)
    avg_pool_5 = GlobalAveragePooling1D()(conv1d_5)
    max_pool_5 = GlobalMaxPooling1D()(conv1d_5)
    conc = concatenate([avg_pool_2, max_pool_2, avg_pool_3, max_pool_3, avg_pool_5, max_pool_5])

    main_output = Dense(category_len, activation='softmax', name='aux_output')(conc)

    model = Model(inputs=main_input, outputs=main_output)
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy",
                  metrics=[metrics.top_k_categorical_accuracy, 'accuracy'])
    # start to train
    print("start *** ")
    start_time = time.time()
    tensorboard = callbacks.TensorBoard(log_dir=f'logs_{category_len}')
    callback_lists = [tensorboard]
    history = model.fit(model_main_data, model_label, epochs=40, batch_size=128,
                        validation_split=0.1, callbacks=callback_lists)
    print(max(history.history['val_acc']))
    print(max(history.history['val_top_k_categorical_accuracy']))
    print("save model")
    spend_time = time.time() - start_time
    print(spend_time)
    model.save("./data/only_cnn.h5")


def only_lstm(model_main_data, model_label):
    print("Building only lstm model *** ")

    main_input = Input(shape=(49, 200), dtype="float32", name="main_input")
    lstm_out = LSTM(256)(main_input)

    main_output = Dense(category_len, activation='softmax', name='aux_output')(lstm_out)

    model = Model(inputs=main_input, outputs=main_output)
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy",
                  metrics=[metrics.top_k_categorical_accuracy, 'accuracy'])
    # start to train
    print("start *** ")
    start_time = time.time()
    tensorboard = callbacks.TensorBoard(log_dir=f'logs_{category_len}')
    callback_lists = [tensorboard]
    history = model.fit(model_main_data, model_label, epochs=60, batch_size=128,
                        validation_split=0.1, callbacks=callback_lists)
    print(max(history.history['val_acc']))
    print(max(history.history['val_top_k_categorical_accuracy']))
    print("save model")
    spend_time = time.time() - start_time
    print(spend_time)
    model.save("./data/only_lstm.h5")


if __name__ == '__main__':
    # parse = argparse.ArgumentParser()
    # parse.add_argument('--model', type=str, default='both', help="train or predict (default True)")

    # get nid embedding
    nid_dict = {}
    category_dict = {}
    cls = 1000

    with open('./data/emb_cntr.txt', 'r') as f:
        for i in f.readlines()[1:]:
            nid_dict[i.split(" ")[0]] = list(map(float, i.split()[1:]))

    for index, k in enumerate(nid_dict.keys()):
        category_dict[k] = index

    # load data.pkl
    data = pickle.load(open('./data/data.pkl', 'rb'))

    main_data = []
    label = []

    # create train data
    for index, d in enumerate(data):
        k, X, y = pre_df(d)
        main_data.append(X)
        label.append(y)

    category_len = max(label) + 1
    y_label = np_utils.to_categorical(np.array(label))

    # lstm_cnn(model_main_data=np.array(main_data), model_label=y_label)
    # only_lstm(model_main_data=np.array(main_data), model_label=y_label)
    only_cnn(model_main_data=np.array(main_data), model_label=y_label)
