# -*- coding: utf-8 -*-
from __future__ import print_function, division
import time
import os
import numpy as np
import yaml
from model import ft_net_dense, ft_net, DisentangleNet
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from model import load_network, save_network, load_whole_network, save_whole_network
from losses import SoftLabelLoss
from datasets import RandomErasing
import shutil
from finetune import generate_cluster_dbscan, generate_cluster_kmeans
from get_multi_target_features import get_features, get_distances
from test import test_function
from evaluate import evaluate_function
from scipy.io import loadmat
from torchvision import datasets, models, transforms

version = torch.__version__

######################################################################
# Options
# --------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--name', default='', type=str, help='output model name')
parser.add_argument('--save_model_name', default='', type=str, help='save_model_name')
parser.add_argument('--data_stage1', default='market', type=str, help='training source dir path')
parser.add_argument('--data_stage2', default='duke', type=str, help='training target dir path')
parser.add_argument('--batchsize', default=8, type=int, help='batchsize')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--net_loss_model', default=1, type=int, help='net_loss_model')
parser.add_argument('--gpu', type=str, default='0', help='GPU id to use.')
parser.add_argument('--margin', default=0.5, type=float, help='margin')
parser.add_argument('--poolsize', default=128, type=int, help='poolsize')

opt = parser.parse_args()
print('opt = %s' % opt)
print('net_loss_model = %d' % opt.net_loss_model)
print('save_model_name = %s' % opt.save_model_name)
dir_name = os.path.join('./model', opt.name)
if not os.path.exists('model'):
    os.mkdir('model')
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
######################################################################
# Load Data
# --------------------------------------------------------------------
#
transform_train_list = [
    transforms.Resize((256, 128), interpolation=3),
    transforms.Pad(10),
    transforms.RandomCrop((256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

if opt.erasing_p > 0:
    transform_train_list = transform_train_list + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]

data_transforms = {
    'train': transforms.Compose(transform_train_list),
}

use_gpu = torch.cuda.is_available()


def get_dataset(stage=1):
    if stage == 1:
        data_path = os.path.join(data_dir_stage1, 'train_all_new')
    else:
        data_path = os.path.join(data_dir_stage2, 'train_all_cluster')
    image_datasets = {}
    image_datasets['train'] = datasets.ImageFolder(data_path,
                         data_transforms['train'])
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                  shuffle=True, num_workers=0) for x in ['train']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
    return dataloaders, dataset_sizes


def get_soft_label_lsr(labels, w_main=0.7, bit_num=6):
    # w_reg is used to prevent data overflow, it a value close to zero
    w_reg = (1.0 - w_main) / bit_num
    if w_reg < 0:
        print('w_main=%s' % (w_main))
        exit()
    soft_label = np.zeros((len(labels), int(bit_num)))
    soft_label.fill(w_reg)
    for i in np.arange(len(labels)):
        soft_label[i][labels[i]] = w_main + w_reg
    return torch.Tensor(soft_label)


def train(model, criterion_identify, optimizer, scheduler, class_num, num_epochs,
          stage=1):
    global setting
    since = time.time()
    best_acc = 0.0
    best_loss = 10000.0
    best_epoch = -1
    r_id = 0.6
    w_main_id = 0.8

    print('r_id = %.2f ' % r_id)
    print('w_main_id = %.2f' % w_main_id)

    dataloaders, dataset_sizes = get_dataset(stage)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, id_labels = data
                now_batch_size, c, h, w = inputs.shape
                if now_batch_size < opt.batchsize:  # next epoch
                    continue

                id_labels_soft = get_soft_label_lsr(id_labels, w_main=w_main_id, bit_num=class_num)

                if use_gpu:
                    inputs = inputs.cuda()
                    id_labels_soft = id_labels_soft.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                output = model(inputs)[0]
                _, id_preds = torch.max(output.detach(), 1)
                loss = r_id * criterion_identify(output, id_labels_soft)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.item()  # * opt.batchsize
                # running_corrects += float(torch.sum(id_preds == id_labels.detach()))
                running_corrects += float(torch.sum(id_preds == id_labels_soft.argmax(1).detach()))

            datasize = dataset_sizes[phase] // opt.batchsize * opt.batchsize
            epoch_loss = running_loss / datasize
            epoch_acc = running_corrects / datasize

            print('{} Loss: {:.4f}  Acc: {:.4f} '.format(phase, epoch_loss, epoch_acc))
            if epoch_acc > best_acc or (np.fabs(epoch_acc - best_acc) < 1e-5 and epoch_loss < best_loss):
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_epoch = epoch
                save_whole_network(model, opt.name, 'best' + '_' + str(opt.net_loss_model))

            if epoch % 10 == 9:
                save_whole_network(model, opt.name, epoch)

    time_elapsed = time.time() - since
    print('best_epoch = %s     best_loss = %s     best_acc = %s' % (best_epoch, best_loss, best_acc))
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    save_whole_network(model, opt.name, 'last' + '_' + str(opt.net_loss_model))
    return model


def initial_model(stage=1):
    if stage == 1:
        class_num = opt.class_base_stage1
    else:
        class_num = opt.class_base_stage2
    print('stage:%d   class_num = %d' % (stage, class_num))
    model = ft_net(id_num=class_num)
    if use_gpu:
        model.cuda()

    # Initialize loss functions
    criterion_identify = SoftLabelLoss()
    classifier_id = list(map(id, model.id_classifier.parameters()))
    classifier_fc = list(map(id, model.fc.parameters()))
    id_params = filter(lambda p: id(p) in classifier_id, model.parameters())
    fc_params = filter(lambda p: id(p) in classifier_fc, model.parameters())
    base_params = filter(lambda p: id(p) not in classifier_id + classifier_fc, model.parameters())

    if stage == 1:
        epoch = 22
        optimizer_ft = optim.SGD([
            {'params': id_params, 'lr': 1 * opt.lr},
            {'params': fc_params, 'lr': 1 * opt.lr},
            {'params': base_params, 'lr': 0.1 * opt.lr},
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[10, 17], gamma=0.1)
    else:
        epoch = 8
        ratio_lr = 0.05
        print('ratio_lr = %.2f' % ratio_lr)
        optimizer_ft = optim.SGD([
            {'params': id_params, 'lr': ratio_lr * 1 * opt.lr},
            {'params': fc_params, 'lr': ratio_lr * 1 * opt.lr},
            {'params': base_params, 'lr': ratio_lr * 0.1 * opt.lr},
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[8], gamma=0.1)
    print('net_loss_model = %s   epoch = %3d' % (opt.net_loss_model, epoch))
    return model, criterion_identify, optimizer_ft, exp_lr_scheduler, class_num, epoch


data_dir_stage1 = os.path.join('data', opt.data_stage1, 'pytorch')
data_dir_stage2 = os.path.join('data', opt.data_stage2, 'pytorch')
stage1_train = True
stage2_train = True
for i in np.arange(1):
    if stage1_train:
        setting = i
        print('setting = %d' % setting)
        print('train stage1 ......')
        opt.class_base_stage1 = len(os.listdir(os.path.join(data_dir_stage1, 'train_all_new')))
        print('opt.class_base_stage1 = %d' % opt.class_base_stage1)
        model, criterion_identify, optimizer_ft, exp_lr_scheduler, class_num, epoch = initial_model(stage=1)
        model = train(model, criterion_identify, optimizer_ft, exp_lr_scheduler, class_num, epoch, stage=1)
        save_whole_network(model, opt.name, 'pretrain')
        test_function(test_dir=opt.data_stage1, net_loss_model=opt.net_loss_model, which_epoch='last')
        evaluate_function()
        test_function(test_dir=opt.data_stage2, net_loss_model=opt.net_loss_model, which_epoch='last')
        evaluate_function()

for k in np.arange(1):
    if stage2_train:
        print('train stage2 ......')
        stage2_since = time.time()
        setting = k
        print('k = %d' % k)
        print('setting = %d' % setting)
        eps0_all = 0.8
        eps0_sid = 0.8
        eps0_did = 0.8
        if 'duke' in data_dir_stage2:
            min_samples = 10
        else:
            min_samples = 8
        iter_finetune_epoch = 10
        for i in np.arange(iter_finetune_epoch):
            print('current: %d    total iter_finetune_epoch: %d' % (i, iter_finetune_epoch))
            cluster_result_path = os.path.join(data_dir_stage2, 'train_all_cluster')
            # #update features
            get_features(order=i, data_dir=data_dir_stage1,
                         net_loss_model=opt.net_loss_model,
                         which_epoch='last')
            get_features(order=i, data_dir=data_dir_stage2,
                         net_loss_model=opt.net_loss_model,
                         which_epoch='last')
            if i == 0:
                dist_all, eps0_all = get_distances(opt.data_stage1, opt.data_stage2, i, ratio=0.003)
            else:
                dist_all, _ = get_distances(opt.data_stage1, opt.data_stage2, i, ratio=0.003)
            ##cluster with DBSCAN
            generate_cluster_dbscan(cluster_result_path, dist=dist_all, eps=eps0_all, min_samples=min_samples,
                             data_dir=opt.data_stage2)
            # generate_cluster_kmeans(cluster_result_path, dist=dist_all, eps=eps0_all, min_samples=min_samples,
            #                  data_dir=opt.data_stage2)
            data_dir_stage2 = os.path.join('data', opt.data_stage2, 'pytorch')
            opt.class_base_stage2 = len(os.listdir(cluster_result_path))
            print('opt.class_base_stage2 = %d' % opt.class_base_stage2)
            model, criterion_identify, optimizer_ft, exp_lr_scheduler, class_num, epoch = initial_model(stage=2)
            if i == 0:
                model = load_whole_network(model, opt.name, 'pretrain')
            else:
                model = load_whole_network(model, opt.name, 'last' + '_' + str(opt.net_loss_model))
            model = train(model, criterion_identify, optimizer_ft, exp_lr_scheduler, class_num, epoch, stage=2)
            save_whole_network(model, opt.name, 'last' + '_' + str(opt.net_loss_model) + '_' + str(i))
            test_function(test_dir=opt.data_stage2, net_loss_model=opt.net_loss_model, which_epoch='last')
            evaluate_function()
            stage2_elapse = time.time() - stage2_since
            print('Finetuning elapse in {:.0f}m {:.0f}s'.format(stage2_elapse // 60, stage2_elapse % 60))
