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

matplotlib.use('agg')
import time
import os
from model import ft_net, save_network
from random_erasing import RandomErasing
import yaml
from torch.utils.data import Dataset, DataLoader
from datasets import TripletFolderPN, TripletFolder
version = torch.__version__

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--name', default='ide_triplet', type=str, help='output model name')
parser.add_argument('--save_model_name', default='', type=str, help='save_model_name')
parser.add_argument('--data_dir', default='market', type=str, help='training dir path')
parser.add_argument('--batchsize', default=16, type=int, help='batchsize')
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

image_datasets['train'] = TripletFolder(os.path.join(data_dir, 'train_all'),
                                  data_transforms['train'])

class_num = len(os.listdir(os.path.join(data_dir, 'train_all')))
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=True, num_workers=8)
               for x in ['train']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}

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

def train(model, criterion_id, criterion_triplet, optimizer, scheduler, num_epochs=25):
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
            id_running_loss = 0.0
            id_running_corrects = 0.0
            triplet_running_loss = 0.0
            triplet_running_corrects = 0.0
            running_margin = 0.0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                anchors, labels, pos = data
                now_batch_size, c, h, w = anchors.shape
                if now_batch_size < opt.batchsize:  # next epoch
                    continue

                pos = pos.view(-1, c, h, w)
                # copy pos labels 4times
                pos_labels = labels.repeat(4).reshape(4, now_batch_size)
                pos_labels = pos_labels.transpose(0, 1).reshape(4 * now_batch_size)
                if use_gpu:
                    anchors = anchors.cuda()
                    pos = pos.cuda()
                    labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                anchor_outputs, anchor_features = model(anchors)
                _, anchor_preds = torch.max(anchor_outputs.detach(), 1)
                loss_id = criterion_id(anchor_outputs, labels)

                pos_outputs, pos_features = model(pos)
                f = anchor_features
                pf = pos_features
                neg_labels = pos_labels
                # hard-neg
                # ----------------------------------
                nf_data = pf  # 128*512
                rand = np.random.permutation(4 * now_batch_size)
                nf_data = nf_data[rand, :]
                neg_labels = neg_labels[rand]
                nf_t = nf_data.transpose(0, 1)  # 512*64
                score = torch.mm(f.data, nf_t)  # cosine 16*64
                score, rank = score.sort(dim=1, descending=True)  # score high == hard
                labels_cpu = labels.cpu()
                nf_hard = torch.zeros(f.shape).cuda()
                for k in range(now_batch_size):
                    hard = rank[k, :]
                    for kk in hard:
                        now_label = neg_labels[kk]
                        anchor_label = labels_cpu[k]
                        if now_label != anchor_label:
                            nf_hard[k, :] = nf_data[kk, :]
                            break

                # hard-pos
                # ----------------------------------
                pf_hard = torch.zeros(f.shape).cuda()  # 16*512
                for k in range(now_batch_size):
                    pf_data = pf[4 * k:4 * k + 4, :]
                    pf_t = pf_data.transpose(0, 1)  # 512*4
                    ff = f.data[k, :].reshape(1, -1)  # 1*512
                    score = torch.mm(ff, pf_t)  # cosine
                    score, rank = score.sort(dim=1, descending=False)  # score low == hard
                    pf_hard[k, :] = pf_data[rank[0][0], :]

                # loss
                # ---------------------------------
                pscore = torch.sum(f * pf_hard, dim=1)
                nscore = torch.sum(f * nf_hard, dim=1)
                loss_triplet = criterion_triplet(f, pf_hard, nf_hard)

                loss = loss_id + 0.2 * loss_triplet

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.item()  # * opt.batchsize
                id_running_loss += loss_id.item()
                id_running_corrects += float(torch.sum(anchor_preds == labels.detach()))
                triplet_running_loss += loss_triplet.item()  # * opt.batchsize
                triplet_running_corrects += float(torch.sum(pscore > nscore + 0.5))
                running_margin += float(torch.sum(pscore - nscore))

            datasize = dataset_sizes[phase] // opt.batchsize * now_batch_size
            epoch_loss = running_loss / datasize
            id_epoch_loss = id_running_loss / datasize
            id_epoch_acc = id_running_corrects / datasize
            triplet_epoch_loss = triplet_running_loss / datasize
            triplet_epoch_acc = triplet_running_corrects / datasize
            epoch_margin = running_margin / datasize
            epoch_acc = (id_epoch_acc + triplet_epoch_acc) / 2.0

            print(
                '{} Loss: {:.4f}  Acc: {:.4f} id_loss: {:.4f}  id_acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc, id_epoch_loss, id_epoch_acc))
            print('{} triplet_epoch_loss: {:.4f} triplet_epoch_acc: {:.4f} MeanMargin: {:.4f}'.format(
                phase, triplet_epoch_loss, triplet_epoch_acc, epoch_margin))

            if epoch_acc > best_acc or (np.fabs(epoch_acc - best_acc) < 1e-5 and epoch_loss < best_loss):
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_epoch = epoch
                save_network(model, name, 'best' + '_' + str(opt.net_loss_model))

            if epoch % 10 == 9:
                save_network(model, name, epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

    print('best_epoch = %s     best_loss = %s     best_acc = %s' % (best_epoch, best_loss, best_acc))
    save_network(model, name, 'last' + '_' + str(opt.net_loss_model))
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
model = ft_net(class_num)
if use_gpu:
    model.cuda()

# print('model structure')
# print(model)

criterion_id = nn.CrossEntropyLoss()
criterion_triplet = nn.TripletMarginLoss(margin=0.5)

classifier_id = list(map(id, model.classifier.parameters())) \
                + list(map(id, model.model.fc.parameters()))
classifier_params = filter(lambda p: id(p) in classifier_id, model.parameters())
base_params = filter(lambda p: id(p) not in classifier_id, model.parameters())

optimizer_ft = optim.SGD([
    {'params': classifier_params, 'lr': 1 * opt.lr},
    {'params': base_params, 'lr': 0.1 * opt.lr},
], weight_decay=5e-4, momentum=0.9, nesterov=True)


epoch = 40
step = 12
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step, gamma=0.1)
print('net_loss_model = %s   epoc = %3d   step = %3d' % (opt.net_loss_model, epoch, step))
model = train(model, criterion_id, criterion_triplet, optimizer_ft, exp_lr_scheduler,
              num_epochs=epoch)


