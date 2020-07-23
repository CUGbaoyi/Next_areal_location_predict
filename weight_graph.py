#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import numpy as np

"""
creating weighted graph by interval
"""


# transform time
def trans_date(d: str):
    n = d.replace(' +0800', '')
    dt_obj = datetime.datetime.strptime(n, "%a %b %d %H:%M:%S %Y")
    date_stamp = dt_obj.timestamp()
    return int(date_stamp)


def trans_weight(uid: str, k: int = 10, cls: int = 100):
    """
    cal the weight of each cluster area
    :param uid: user id
    :param k: weight k default is 10
    :param cls: class number
    :return:
    """

    def logistic(x):
        """
        logistic
        :param x:
        :return:
        """
        return 1 / (1 + np.exp(-x / k))

    # first of all, find the cluster area a user had visited. Here is a example
    p_cur.execute(f"select created_at, cntr_id from wuhan.wuhan_cls_{cls} where uid = '{uid}'")
    time_list = sorted([(trans_date(t[0]), t[1]) for t in p_cur.fetchall()], key=lambda x: x[0])
    print(time_list)

    # get the max time interval
    max_time_interval = time_list[-1][0] - time_list[1][0]
    handle_time_list = []

    for t in time_list:
        if t[1] not in handle_time_list:
            handle_time_list.append(t)

    # handle edge data
    for i in range(len(handle_time_list)):
        # There is no edge, then create new one
        if handle_time_list[i][1] in edge:
            pass
        else:
            edge[handle_time_list[i][1]] = {}

        for j in range(i + 1, len(handle_time_list)):
            nid_x = handle_time_list[i][1]
            time_x = handle_time_list[i][0]
            nid_y = handle_time_list[j][1]
            time_y = handle_time_list[j][0]

            if edge[nid_x].get(nid_y) is not None:
                edge[nid_x][nid_y] += logistic(max_time_interval / (time_y - time_x))
                print("$" * 50)
                print(edge[nid_x][nid_y])
            else:
                edge[nid_x][nid_y] = logistic(max_time_interval / (time_y - time_x))
                print(edge[nid_x][nid_y])


if __name__ == '__main__':

    # save the edge data
    edge = {}
    result = []

    # you should create you own uid list
    uid_list = []
    cls = 1000

    for uid in uid_list:
        try:
            trans_weight(uid, 10, cls)
        except Exception as e:
            print(e)

    for k, v in edge.items():
        for child_k, weight in v.items():
            result.append(" ".join((str(k), str(child_k), str(weight))))

    with open(f"./{cls}/data/cntr_weight_{cls}.csv", 'a', encoding="utf-8") as f:
        for d in result:
            f.write(d + '\n')
