import numpy as np
import os
import shutil
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import savemat
import time
import math
from sklearn.cluster import DBSCAN, KMeans
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import torch
import argparse
from collections import Counter


def analysis_features():
    m = loadmat('duke' + '_pytorch_target_result.mat')
    print('type(m) = %s' % type(m))
    print(m.keys())
    target_features = m['train_f']
    data_num = len(target_features)

    np.random.seed(100)

    print('len(target_features) = %s' % len(target_features))
    print('target_features[0].size = %d' % target_features[0].size)
    # savemat(os.path.join(dst_path, 'target_features.mat'), {'features': target_features})

    target_labels = m['train_label'][0]
    # print('target_labels = %s' % (target_labels))
    # print('unique lable = %s' % np.unique(np.sort(target_labels)))
    print('real class_num = %s' % len(np.unique(np.sort(target_labels))))
    print('len(target_labels) = %s' % len(target_labels))

    target_names = m['train_name']
    print(sorted(Counter(target_labels).values())[:10])
    print(sorted(Counter(target_labels).values())[-10:])

    indices = np.random.permutation(len(target_labels))
    target_labels = target_labels[indices]
    target_features = target_features[indices]

    same_dist = []
    diff_dist = []
    same_max = []
    diff_min = []
    same_avg = []
    diff_avg = []

    for i in np.arange(len(target_features)):
        dist = ((target_features[i] - target_features) * (target_features[i] - target_features)).sum(1)
        same_sub = [dist[j] for j in np.arange(len(target_features)) if target_labels[j] == target_labels[i]]
        diff_sub = [dist[j] for j in np.arange(len(target_features)) if target_labels[j] != target_labels[i]]
        same_dist.append(same_sub)
        diff_dist.append(diff_sub[: len(same_sub)])
        same_max.append(np.max(same_sub))
        diff_min.append(np.min(diff_sub))
        same_avg.append(np.sum(same_sub) / (len(same_sub) - 1 + 1e-6))
        diff_avg.append(np.mean(diff_sub))
        if i % 200 == 0:
            print('i = %3d' % i)
        if i > 1000:
            break

    cnt = np.sum(np.array(same_max) < np.array(diff_min))
    ratio = cnt / len(same_max)
    print(ratio)
    print('avg same_max = %.3f    avg diff_min = %.3f' % (np.mean(same_max), np.mean(diff_min)))
    print('avg same = %.3f    avg diff = %.3f' % (np.mean(same_avg), np.mean(diff_avg)))


##############################################################################
# Compute DBSCAN

def generate_cluster_dbscan(cluster_result_path, dist=None, eps=0.8, min_samples=10, data_dir=None):
    m = loadmat(data_dir + '_pytorch_target_result.mat')
    target_features = m['train_f']
    data_num = len(target_features)
    print('len(target_features) = %s' % len(target_features))
    print('target_features[0].size = %d' % target_features[0].size)
    target_labels = m['train_label'][0]
    print('real class_num = %s' % len(np.unique(np.sort(target_labels))))
    print('len(target_labels) = %s' % len(target_labels))
    target_names = m['train_name']

    if os.path.exists(cluster_result_path):
        shutil.rmtree(cluster_result_path)
    os.mkdir(cluster_result_path)
    process_num = m['train_label'][0].shape[0]
    if dist is None:
        X = target_features[:process_num]
    else:
        X = dist[:process_num]
    labels_true = target_labels[:process_num]
    names = target_names[:process_num]
    print('DBSCAN starting ......')
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    print(sorted(Counter(labels).values())[:10])
    print(sorted(Counter(labels).values())[-10:])

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)
    n_node = np.sum(labels != -1)
    print('Estimated number of nodes: %d' % n_node)
    dir_cnt = 0
    image_cnt = 0
    for i in np.arange(n_clusters_):
        files = names[np.where(labels == i)]
        if len(files) > np.max((2, min_samples - 1)):
            dir_path = os.path.join(cluster_result_path, str(i).zfill(4))
            os.mkdir(dir_path)
            for file in files:
                file = file.strip()
                shutil.copy(
                    os.path.join(os.path.split(os.path.split(cluster_result_path)[0])[0], 'bounding_box_train', file),
                    os.path.join(dir_path, file))
        dir_cnt += 1
        image_cnt += len(files)

    print('valid cluster number: %d    file number:%d' % (dir_cnt, image_cnt))
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    # print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
    # print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))

def generate_cluster_kmeans(cluster_result_path, dist=None, eps=0.8, min_samples=10, data_dir=None):
    m = loadmat(data_dir + '_pytorch_target_result.mat')
    target_features = m['train_f']
    data_num = len(target_features)
    print('len(target_features) = %s' % len(target_features))
    print('target_features[0].size = %d' % target_features[0].size)
    target_labels = m['train_label'][0]
    print('real class_num = %s' % len(np.unique(np.sort(target_labels))))
    print('len(target_labels) = %s' % len(target_labels))
    target_names = m['train_name']

    if os.path.exists(cluster_result_path):
        shutil.rmtree(cluster_result_path)
    os.mkdir(cluster_result_path)
    process_num = m['train_label'][0].shape[0]
    if dist is None:
        X = target_features[:process_num]
    else:
        X = dist[:process_num]
    labels_true = target_labels[:process_num]
    names = target_names[:process_num]
    print('KMeans starting ......')
    db = KMeans(n_clusters=702).fit(X)
    labels = db.labels_
    print(sorted(Counter(labels).values())[:10])
    print(sorted(Counter(labels).values())[-10:])

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)
    n_node = np.sum(labels != -1)
    print('Estimated number of nodes: %d' % n_node)
    dir_cnt = 0
    image_cnt = 0
    for i in np.arange(n_clusters_):
        files = names[np.where(labels == i)]
        if len(files) > np.max((2, min_samples - 1)):
            dir_path = os.path.join(cluster_result_path, str(i).zfill(4))
            os.mkdir(dir_path)
            for file in files:
                file = file.strip()
                shutil.copy(
                    os.path.join(os.path.split(os.path.split(cluster_result_path)[0])[0], 'bounding_box_train', file),
                    os.path.join(dir_path, file))
        dir_cnt += 1
        image_cnt += len(files)

    print('valid cluster number: %d    file number:%d' % (dir_cnt, image_cnt))
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    # print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
    # print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))


if __name__ == '__main__':
    # analysis_features()
    # cluster()
    # generate_cluster_data('data/duke/pytorch/train_all_cluster')
    generate_cluster('data/duke/pytorch/train_all_cluster')
