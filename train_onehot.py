#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from keras.layers import Input, LSTM, Dense, Masking, concatenate, Bidirectional, Conv1D, GlobalMaxPooling1D, \
    GlobalAveragePooling1D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import np_utils
from keras import metrics, regularizers, callbacks
import numpy as np
import pickle
import time


def list_norm(l: list):
    """
    data norm
    :param l:
    :return:
    """
    return [(i - min(l)) / (max(l) - min(l)) for i in l]


def to_one_hot(n, num):
    """
    data to one_hot
    :param n:
    :param num:
    :return:
    """
    p = [0] * num
    p[n] = 1
    return p


def numpy_handle_train_df():
    """
    get one hot train data
    :return:
    """

    def parse_time(x):
        """
        :param x:
        :return:
        """
        date_total = []
        date_uid = int(x['uid'])
        date_cntr = str(x['cntr_id'])

        date_month = [0] * 12
        date_hour = [0] * 24
        date_week = [0] * 7

        date_month[int(x['month']) - 1] = 1
        date_hour[int(x['hour'])] = 1
        date_week[int(x['week'])] = 1

        date_total.append(date_uid)
        date_total.append(date_cntr)
        date_total.extend(date_month + date_hour + date_week)

        return pd.Series(date_total)

    nid_list = df['cntr_id'].tolist()
    uid_list = df['uid'].tolist()
    cntr = []
    total_list = []

    for n in nid_list:
        cntr.append(one_hot[str(n)])

    cntr_df = pd.DataFrame(cntr)

    print(cntr_df.shape)
    time_df = df.apply(parse_time, axis=1)
    concat_df = pd.concat([time_df, cntr_df], axis=1)
    concat_df.columns = list(range(0, cntr_num + 45))

    for uid in set(uid_list):
        tem = concat_df[concat_df[0] == int(uid)]
        if tem.shape[0] > 2:
            tem.pop(0)
            total_list.append({uid: tem})

    # the train data will save in data.pkl
    with open('./data/data.pkl', 'wb') as output:
        pickle.dump(total_list, output)

    return total_list


def pre_df(_d, cntr_num):
    """
    pre train
    :param _d:
    :param cntr_num
    :return: (uid_id, lstm x, lstm y, valid y, aux data(time information which are not used in this task))
    """
    for uid_index, v in _d.items():
        v = np.array(v)
        # split the data to X and Y
        pre_X, y = v[:-2, 44:], category_dict[v[-2, 0]]

        # flatten a
        p = pre_X.flatten()
        # get nodata count, ready to padding
        padding_weight = (48 * (cntr_num) - p.shape[0])
        # padding and reshape to 49 * 200
        X = np.pad(p, (0, padding_weight), 'constant', constant_values=-1.).reshape(49, cntr_num)

        return uid_index, X, y


def build_model(model_main_data, model_label, category_len, cntr_num):
    """
    build model
    :return:
    """
    print("Building one hot model")

    main_input = Input(shape=(48, cntr_num), dtype="float32", name="main_input")

    # build model
    lstm_out = Bidirectional(LSTM(256, return_sequences=True))(main_input)
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

    print("start *** ")
    start_time = time.time()
    tensorboard = callbacks.TensorBoard(log_dir=f'logs_{category_len}')
    early_stopping = callbacks.EarlyStopping(monitor='val_acc', patience=10)
    callback_lists = [tensorboard, early_stopping]
    history = model.fit(model_main_data, model_label, epochs=150, batch_size=128,
                        validation_split=0.1, callbacks=callback_lists)
    print(max(history.history['val_acc']))
    print(max(history.history['val_top_k_categorical_accuracy']))
    print("save model")
    spend_time = time.time() - start_time
    print(spend_time)
    model.save("./data/_model1.h5")


if __name__ == '__main__':
    # get nid embedding
    nid_dict = {}
    category_dict = {}
    one_hot = {}
    cls = 1000

    with open('./data/emb_cntr.txt', 'r') as f:
        for i in f.readlines()[1:]:
            nid_dict[i.split(" ")[0]] = list(map(float, i.split()[1:]))

    cntr_num = len(nid_dict.keys())

    for index, k in enumerate(nid_dict.keys()):
        category_dict[k] = index
        one_hot[k] = to_one_hot(index, cntr_num)

    df = pd.read_csv(f"./data/wuhan_final_data_{cls}.csv")
    df.drop(['mid', 'created_at'], axis=1, inplace=True)

    # data pkl
    print(cntr_num)
    # data = pickle.load(open('./data/data.pkl', 'rb'))
    data = pickle.load(open('./data/data.pkl', 'rb'))

    main_data = []
    auxiliary_data = []
    label = []
    _label = []

    for index, d in enumerate(data):
        k, X, y = pre_df(d, cntr_num)
        main_data.append(X)
        label.append(y)

    category_len = max(label) + 1
    y_label = np_utils.to_categorical(np.array(label))

    build_model(model_main_data=np.array(main_data), model_label=y_label,
                category_len=category_len, cntr_num=cntr_num)
