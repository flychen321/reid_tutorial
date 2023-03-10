# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import os
import scipy.io
import yaml
import torch
from model import ft_net, DisentangleNet
from model import load_network, load_whole_network
import numpy as np
from datasets import ChannelTripletFolder, ChannelDatasetAllDomain
from rerank_for_cluster import re_ranking
from scipy.io import loadmat
from scipy.io import savemat
from torchvision import datasets, models, transforms

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_feature(model, dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        ff = torch.FloatTensor(n, 512).zero_().cuda()
        for i in range(2):
            if (i == 1):
                img = fliplr(img)
            input_img = img.cuda()
            outputs = model(input_img)
            ff = ff + outputs[1]
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        ff = ff.detach().cpu().float()
        features = torch.cat((features, ff), 0)
    return features


def get_id(img_path, test_dir):
    camera_id = []
    labels = []
    names = []
    for path, v in img_path:
        # filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        if 'msmt' in test_dir:
            camera = filename[9:11]
        else:
            camera = filename.split('c')[1][0]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera))
        names.append(filename)
    return camera_id, labels, names


def get_features(order=0, data_dir=None, net_loss_model=None, which_epoch=None):
    ######################################################################
    # Load Data
    # --------------------------------------------------------------------
    data_transforms = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset_list = ['train_all_new']
    image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in dataset_list}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                                  shuffle=False, num_workers=0) for x in dataset_list}
    use_gpu = torch.cuda.is_available()
    print('test_dir = %s   net_loss_model = %d' % (data_dir, net_loss_model))
    dataset_path = []
    for i in range(len(dataset_list)):
        dataset_path.append(image_datasets[dataset_list[i]].imgs)
    dataset_cam = []
    dataset_label = []
    dataset_name = []
    for i in range(len(dataset_list)):
        cam, label, n = get_id(dataset_path[i], data_dir)
        dataset_cam.append(cam)
        dataset_label.append(label)
        dataset_name.append(n)

    ######################################################################
    # Load Collected data Trained model
    print('---------test-----------')
    class_num = len(os.listdir(os.path.join(data_dir, 'train_all_new')))
    model = ft_net(id_num=class_num)
    if use_gpu:
        model.cuda()

    name = ''
    if order == 0:
        model = load_whole_network(model, name, 'pretrain')
    else:
        if 'best' in which_epoch or 'last' in which_epoch:
            model = load_whole_network(model, name, which_epoch + '_' + str(net_loss_model))
        else:
            model = load_whole_network(model, name, which_epoch)
    model = model.eval()
    if use_gpu:
        model = model.cuda()

    with torch.no_grad():
            # Extract feature
            dataset_feature = []
            dataset_feature.append(extract_feature(model, dataloaders[dataset_list[0]]))
            result = {'train_f': dataset_feature[0].numpy(), 'train_label': dataset_label[0],
                      'train_cam': dataset_cam[0],
                      'train_name': dataset_name[0]}
            scipy.io.savemat(data_dir.split('/')[1] + '_pytorch_target_result.mat', result)

def intra_distance(features):
    x = features
    y = features
    """
    get the Euclidean Distance between to matrix
    (x-y)^2 = x^2 + y^2 - 2xy
    :param x:
    :param y:
    :return:
    """
    (rowx, colx) = x.shape
    (rowy, coly) = y.shape
    if colx != coly:
        raise RuntimeError('colx must be equal with coly')
    xy = np.dot(x, y.T)
    x2 = np.repeat(np.reshape(np.sum(np.multiply(x, x), axis=1), (rowx, 1)), repeats=rowy, axis=1)
    y2 = np.repeat(np.reshape(np.sum(np.multiply(y, y), axis=1), (rowy, 1)), repeats=rowx, axis=1).T
    dist = x2 + y2 - 2 * xy
    return dist

def get_distances(src_path, tgt_path, order=-1, ratio=0.003):
    print('Calculating feature distances...')
    m = loadmat(src_path + '_pytorch_target_result.mat')
    source_features = m['train_f']
    m = loadmat(tgt_path + '_pytorch_target_result.mat')
    target_features = m['train_f']
    rerank_dist = re_ranking(source_features, target_features, lambda_value=0.1)
    # rerank_dist = intra_distance(target_features)
    # DBSCAN cluster
    tri_mat = np.triu(rerank_dist, 1)  # tri_mat.dim=2
    tri_mat = tri_mat[np.nonzero(tri_mat)]  # tri_mat.dim=1
    tri_mat = np.sort(tri_mat, axis=None)
    top_num = np.round(ratio * tri_mat.size).astype(int)
    eps = tri_mat[:top_num].mean()
    print('eps in cluster: %.3f' % eps)
    eps_list = [0]
    for i in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
        print('i = %5d   %.3f' % (i, tri_mat[:int(0.001 * i * tri_mat.size)].mean()))
        eps_list.append(tri_mat[:int(0.001 * i * tri_mat.size)].mean())
    return rerank_dist, eps


if __name__ == '__main__':
    get_features(order=0, data_dir='data/market/pytorch', net_loss_model=1, which_epoch='last')
