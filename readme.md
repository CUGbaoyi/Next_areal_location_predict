# A-BiLSTM-CNN-model-for-predicting-users-next-areal-location

## Introduction
 We proposed a location prediction model for mining geotagged social media data at an areal scale. In our model, we use [HiSpatialCluster](https://github.com/lopp2005/HiSpatialCluster) algorithm to cluster the whole check-in data and construct a relationship graph of cluster areas. Then we use the [LINE](https://github.com/tangjianpku/LINE) to get the representation vector of the Cluster Areas which can effectively represent the correlation between different locations. Finally, BiLSTM-CNN is used for location prediction.

## How to use
First of all, you should use [HiSpatialCluster](https://github.com/lopp2005/HiSpatialCluster) to cluster the whole data (You can choose a suitable cluster size according to you task). After that, you have got the relationship between users and cluster-id (an example in [wuhan_final_data.csv]. Then Build the graph by the time interval of users visiting different areas (example in [cntr_weight.csv]). Then use [LINE](https://github.com/tangjianpku/LINE) to get the representation vector of the Cluster Areas [emb_cntr.csv]. Finally, use [BiLSTM-CNN] model to predict Cluster Areas.

## Code description
**_weight_graph.py_** shows how we create the Cluster Area graph
**_preprocess.py_** is used to create training data
**_train_model.py_** is the model we used and other comparative models that based on embedding data
**_train_onehot.py_** is the model we used based on OneHot

