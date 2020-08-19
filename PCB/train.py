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
from model import ft_net, save_network, save_whole_network, PCB
from random_erasing import RandomErasing
import yaml
from torch.utils.data import Dataset, DataLoader
from losses import SoftLabelLoss
version = torch.__version__

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--name', default='ide', type=str, help='output model name')
parser.add_argument('--save_model_name', default='', type=str, help='save_model_name')
parser.add_argument('--data_dir', default='market', type=str, help='training dir path')
parser.add_argument('--batchsize', default=48, type=int, help='batchsize')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--alpha', default=1.0, type=float, help='alpha')
parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--net_loss_model', default=0, type=int, help='net_loss_model')
parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50')


opt = parser.parse_args()
opt.PCB = True
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
train_path = os.path.join(data_dir, 'train_all')
image_datasets['train'] = datasets.ImageFolder(train_path,
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

def train(model, criterion, optimizer, scheduler, num_epochs=25):
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
                if opt.PCB:
                    part, feature = model(inputs)
                    sm = nn.Softmax(dim=1)
                    num_part = 6
                    score = sm(part[:,:,0]) + sm(part[:,:,1]) + sm(part[:,:,2]) + sm(part[:,:,3]) + sm(part[:,:,4]) + sm(part[:,:,5])
                    _, id_preds = torch.max(score.data, 1)
                    loss = criterion(part[:,:,0], id_labels)
                    for i in range(num_part - 1):
                        loss += criterion(part[:,:,i + 1], id_labels)
                else:
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
model = PCB(class_num, train=True)
if use_gpu:
    model.cuda()

# print('model structure')
# print(model)
criterion = nn.CrossEntropyLoss()
criterion_soft = SoftLabelLoss()

classifier_id = (list(map(id, model.classifier0.parameters()))
                 +list(map(id, model.classifier1.parameters()))
                 +list(map(id, model.classifier2.parameters()))
                 +list(map(id, model.classifier3.parameters()))
                 +list(map(id, model.classifier4.parameters()))
                 +list(map(id, model.classifier5.parameters()))
                  )
classifier_params = filter(lambda p: id(p) in classifier_id, model.parameters())
base_params = filter(lambda p: id(p) not in classifier_id, model.parameters())
optimizer_ft = optim.SGD([
    {'params': classifier_params, 'lr': 1 * opt.lr},
    {'params': base_params, 'lr': 0.1 * opt.lr},
], weight_decay=5e-4, momentum=0.9, nesterov=True)

epoch = 130
step = 40
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step, gamma=0.1)
print('net_loss_model = %s   epoc = %3d   step = %3d' % (opt.net_loss_model, epoch, step))
model = train(model, criterion, optimizer_ft, exp_lr_scheduler,
              num_epochs=epoch)

# model = train_with_softlabel(model, criterion_soft, optimizer_ft, exp_lr_scheduler,
#               num_epochs=epoch)
