# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib
from model import ft_net, DisentangleNet
from model import load_network, save_network, load_whole_network, save_whole_network
matplotlib.use('agg')
import time
import os
from model import ft_net, save_network, save_whole_network
from random_erasing import RandomErasing
import yaml
from torch.utils.data import Dataset, DataLoader
from losses import SoftLabelLoss
from datasets import Attribute_Dataset
version = torch.__version__

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--name', default='ide', type=str, help='output model name')
parser.add_argument('--save_model_name', default='', type=str, help='save_model_name')
parser.add_argument('--data_dir', default='market1', type=str, help='training dir path')
parser.add_argument('--batchsize', default=48, type=int, help='batchsize')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--alpha', default=1.0, type=float, help='alpha')
parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--net_loss_model', default=0, type=int, help='net_loss_model')

opt = parser.parse_args()
print('opt = %s' % opt)
print('net_loss_model = %d' % opt.net_loss_model)
print('save_model_name = %s' % opt.save_model_name)
data_dir = os.path.join('data', opt.data_dir, 'pytorch')
name = opt.name

######################################################################
# Load Data
# ---------
#

transform_train_list = [
    transforms.Resize((256, 128), interpolation=3),
    transforms.Pad(10),
    transforms.RandomCrop((256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_val_list = [
    transforms.Resize(size=(256, 128), interpolation=3),  # Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

if opt.erasing_p > 0:
    transform_train_list = transform_train_list + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose(transform_train_list)
}


image_datasets = {}
train_id_path = os.path.join(data_dir, 'train_all_c')
image_datasets['train'] = datasets.ImageFolder(train_id_path,
                                  data_transforms['train'])

train_attribute_path = os.path.join(data_dir, 'train_attribute')
image_datasets['attribute'] = Attribute_Dataset(train_attribute_path,
                                  data_transforms['train'])

class_num = len(os.listdir(os.path.join(data_dir, 'train_all_c')))
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=True, num_workers=8)
               for x in ['train', 'attribute']}


dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'attribute']}

use_gpu = torch.cuda.is_available()

since = time.time()
print(time.time() - since)

def get_soft_label_lsr(labels, w_main=0.8, bit_num=751):
    w_reg = (1.0 - w_main) / bit_num
    soft_label = np.zeros((len(labels), int(bit_num)))
    soft_label.fill(w_reg)
    for i in np.arange(len(labels)):
        soft_label[i][labels[i]] = w_main + w_reg
    return torch.Tensor(soft_label)

######################################################################
# Training the model
# ------------------

def train_id(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_acc = 0.0
    best_loss = 10000.0
    best_epoch = -1
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

                if use_gpu:
                    inputs = inputs.cuda()
                    id_labels = id_labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                output = model(inputs)[0]
                _, id_preds = torch.max(output.detach(), 1)
                loss = criterion(output, id_labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.item()  # * opt.batchsize
                running_corrects += float(torch.sum(id_preds == id_labels.detach()))

            datasize = dataset_sizes[phase] // opt.batchsize * opt.batchsize
            epoch_loss = running_loss / datasize
            epoch_acc = running_corrects / datasize

            print('{} Loss: {:.4f}  Acc: {:.4f} '.format(phase, epoch_loss, epoch_acc))
            if epoch_acc > best_acc or (np.fabs(epoch_acc - best_acc) < 1e-5 and epoch_loss < best_loss):
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_epoch = epoch
                save_whole_network(model, name, 'best' + '_' + str(opt.net_loss_model))

            if epoch % 10 == 9:
                save_whole_network(model, name, epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

    print('best_epoch = %s     best_loss = %s     best_acc = %s' % (best_epoch, best_loss, best_acc))
    save_whole_network(model, name, 'last' + '_' + str(opt.net_loss_model))
    save_whole_network(model, name, 'last_ID' + '_' + str(opt.net_loss_model))
    return model

def train_attribute(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_acc = 0.0
    best_loss = 10000.0
    best_epoch = -1
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['attribute']:
            scheduler.step()
            model.train(True)  # Set model to training mode

            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, id_labels = data
                # now_batch_size, c, h, w = inputs.shape
                # if now_batch_size < opt.batchsize:  # next epoch
                #     continue

                if use_gpu:
                    inputs = inputs.cuda()
                    id_labels = id_labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                output = model(inputs)[2]
                _, id_preds = torch.max(output.detach(), 1)
                loss = criterion(output, id_labels)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()
                # statistics
                running_loss += loss.item()  # * opt.batchsize
                running_corrects += float(torch.sum(id_preds == id_labels.detach()))

            datasize = dataset_sizes[phase]
            epoch_loss = running_loss / datasize
            epoch_acc = running_corrects / datasize
            print('did')
            print(model.did_embedding_net.fc.state_dict()['add_block.0.weight'][0, :8])
            print(model.did_embedding_net.model.conv1.weight[0,0,0,:5])
            print('sid')
            print(model.sid_embedding_net.fc.state_dict()['add_block.0.weight'][0, :8])
            print(model.sid_embedding_net.model.conv1.weight[0, 0, 0, :5])
            print('{} Loss: {:.4f}  Acc: {:.4f} '.format(phase, epoch_loss, epoch_acc))
            if epoch_acc > best_acc or (np.fabs(epoch_acc - best_acc) < 1e-5 and epoch_loss < best_loss):
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_epoch = epoch
                save_whole_network(model, name, 'best' + '_' + str(opt.net_loss_model))

            if epoch % 10 == 9:
                save_whole_network(model, name, epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

    print('best_epoch = %s     best_loss = %s     best_acc = %s' % (best_epoch, best_loss, best_acc))
    save_whole_network(model, name, 'last' + '_' + str(opt.net_loss_model))
    save_whole_network(model, name, 'last_attribute' + '_' + str(opt.net_loss_model))
    return model

def train_with_softlabel(model, criterion_soft, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_acc = 0.0
    best_loss = 10000.0
    best_epoch = -1
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

                id_labels_soft = get_soft_label_lsr(id_labels)

                if use_gpu:
                    inputs = inputs.cuda()
                    id_labels_soft = id_labels_soft.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                output = model(inputs)[0]
                _, id_preds = torch.max(output.detach(), 1)
                loss = criterion_soft(output, id_labels_soft)

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
                save_whole_network(model, name, 'best' + '_' + str(opt.net_loss_model))

            if epoch % 10 == 9:
                save_whole_network(model, name, epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

    print('best_epoch = %s     best_loss = %s     best_acc = %s' % (best_epoch, best_loss, best_acc))
    save_whole_network(model, name, 'last' + '_' + str(opt.net_loss_model))
    return model


######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU.
#
dir_name = os.path.join('./model', name)
if not os.path.exists('model'):
    os.mkdir('model')

print('class_num = %d' % (class_num))
id_embedding_net = ft_net(class_num)
attribute_embedding_net = ft_net(6)
model = DisentangleNet(id_embedding_net, attribute_embedding_net)
if use_gpu:
    model.cuda()

# print('model structure')
# print(model)

criterion = nn.CrossEntropyLoss()
criterion_soft = SoftLabelLoss()


# did_classifier_id = list(map(id, model.did_embedding_net.id_classifier.parameters()))
# did_classifier_fc = list(map(id, model.did_embedding_net.fc.parameters()))
# did_id_params = filter(lambda p: id(p) in did_classifier_id, model.parameters())
# did_fc_params = filter(lambda p: id(p) in did_classifier_fc, model.parameters())
# did_base_params = filter(lambda p: id(p) not in did_classifier_id + did_classifier_fc, model.did_embedding_net.parameters())
#
#
# sid_classifier_id = list(map(id, model.sid_embedding_net.id_classifier.parameters()))
# sid_classifier_fc = list(map(id, model.sid_embedding_net.fc.parameters()))
# sid_id_params = filter(lambda p: id(p) in sid_classifier_id, model.parameters())
# sid_fc_params = filter(lambda p: id(p) in sid_classifier_fc, model.parameters())
# sid_base_params = filter(lambda p: id(p) not in sid_classifier_id + sid_classifier_fc, model.sid_embedding_net.parameters())
#
# optimizer_ft = optim.SGD([
#     {'params': did_id_params, 'lr': 1 * opt.lr},
#     {'params': sid_id_params, 'lr': 0.0},
#     {'params': did_fc_params, 'lr': 1 * opt.lr},
#     {'params': sid_fc_params, 'lr': 0.0},
#     {'params': did_base_params, 'lr': 0.1 * opt.lr},
#     {'params': sid_base_params, 'lr': 0.0},
# ], weight_decay=5e-4, momentum=0.9, nesterov=True)
#
# epoch = 130
# step = 40
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step, gamma=0.1)
# print('ID  net_loss_model = %s   epoc = %3d   step = %3d' % (opt.net_loss_model, epoch, step))
# model = train_id(model, criterion, optimizer_ft, exp_lr_scheduler,
#               num_epochs=epoch)


# model = load_whole_network(model, name, 'last_ID_0')


did_classifier_id = list(map(id, model.did_embedding_net.id_classifier.parameters()))
did_classifier_fc = list(map(id, model.did_embedding_net.fc.parameters()))
did_id_params = filter(lambda p: id(p) in did_classifier_id, model.parameters())
did_fc_params = filter(lambda p: id(p) in did_classifier_fc, model.parameters())
did_base_params = filter(lambda p: id(p) not in did_classifier_id + did_classifier_fc, model.did_embedding_net.parameters())


sid_classifier_id = list(map(id, model.sid_embedding_net.id_classifier.parameters()))
sid_classifier_fc = list(map(id, model.sid_embedding_net.fc.parameters()))
sid_id_params = filter(lambda p: id(p) in sid_classifier_id, model.parameters())
sid_fc_params = filter(lambda p: id(p) in sid_classifier_fc, model.parameters())
sid_base_params = filter(lambda p: id(p) not in sid_classifier_id + sid_classifier_fc, model.sid_embedding_net.parameters())

lr_ratio_attribute = 0.1
optimizer_ft = optim.SGD([
    {'params': did_id_params, 'lr': 0.0},
    {'params': sid_id_params, 'lr': lr_ratio_attribute * 1 * opt.lr},
    {'params': did_fc_params, 'lr': 0.0},
    {'params': sid_fc_params, 'lr': lr_ratio_attribute * 1 * opt.lr},
    {'params': did_base_params, 'lr': 0.0},
    {'params': sid_base_params, 'lr': lr_ratio_attribute * 0.1 * opt.lr},
], weight_decay=5e-4, momentum=0.9, nesterov=True)
epoch = 60
step = 25
# epoch = 20
# step = 15
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step, gamma=0.1)
print('Attribute net_loss_model = %s   epoc = %3d   step = %3d' % (opt.net_loss_model, epoch, step))
model = train_attribute(model, criterion, optimizer_ft, exp_lr_scheduler,
              num_epochs=epoch)

# model = train_with_softlabel(model, criterion_soft, optimizer_ft, exp_lr_scheduler,
#               num_epochs=epoch)
