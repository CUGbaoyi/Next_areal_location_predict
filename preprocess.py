#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import pickle


def list_norm(l: list):
    """
    :param l:
    :return:
    """
    return [(i - min(l)) / (max(l) - min(l)) for i in l]


def numpy_handle_train_df():
    """
    get train data
    :return:
    """

    def parse_time(x):
        """
        handle the time information()
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
        cntr.append(nid_dict[str(n)])

    cntr_df = pd.DataFrame(cntr)

    time_df = df.apply(parse_time, axis=1)
    concat_df = pd.concat([time_df, cntr_df], axis=1)
    concat_df.columns = list(range(0, 245))
    _ = concat_df[list(range(2, 245))].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    final_df = pd.concat([concat_df[[0, 1]], _], axis=1)

    for uid in set(uid_list):
        tem = final_df[final_df[0] == int(uid)]
        if tem.shape[0] > 2:
            tem.pop(0)
            total_list.append({uid: tem})

    # the train data will save in data.pkl
    with open('./data/data.pkl', 'wb') as p:
        pickle.dump(total_list, p)

    return total_list


if __name__ == '__main__':
    # get nid embedding result
    nid_dict = {}
    category_dict = {}
    cls = 2000

    with open('./data/emb_cntr.txt', 'r') as f:
        for i in f.readlines()[1:]:
            nid_dict[i.split(" ")[0]] = list(map(float, i.split()[1:]))

    for index, k in enumerate(nid_dict.keys()):
        category_dict[k] = index

    df = pd.read_csv(f"./data/wuhan_final_data_{cls}.csv")
    df.drop(['mid', 'created_at'], axis=1, inplace=True)

    data = numpy_handle_train_df()
