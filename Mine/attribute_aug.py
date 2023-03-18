import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.io import savemat
import time
import random
import os
import math
import cv2
import shutil
import torch
import argparse

# image size: 128 * 64 *3
parser = argparse.ArgumentParser(description='Augment')
parser.add_argument('--data_dir', default='market1', type=str, help='data_dir')
parser.add_argument('--mode', default=5, type=int, help='mode')
opt = parser.parse_args()
print('opt = %s' % opt)
data_dir = opt.data_dir
print('data_dir = %s' % data_dir)
print('opt.mode = %s' % opt.mode)
path = os.path.join('data', data_dir, 'pytorch/train_attribute_original')
dst_path = os.path.join('data', data_dir, 'pytorch/train_attribute')
print('path = %s    dst_path = %s' % (path, dst_path))


def augment_more(max_id_num=200):
    dirs1 = os.listdir(path)
    dirs2 = os.listdir(path)
    dir_len = len(dirs2)
    name = []
    new_dir_num = 0
    new_file_num = 0
    epoc = 0
    head = int(256/5.0)
    body = int(256-head)
    while new_dir_num < max_id_num and epoc < 100:
        epoc += 1
        np.random.shuffle(dirs2)
        for i in range(int(len(dirs2))):
            dir_name = dirs1[i] + dirs2[i]
            files1 = os.listdir(os.path.join(path, dirs1[i]))
            files2 = os.listdir(os.path.join(path, dirs2[i]))

            num = min(len(files1), len(files2))
            if num < 3 or dir_name in name or dirs1[i] == dirs2[i]:
                continue
            index1 = np.random.permutation(len(files1))[:num]
            index2 = np.random.permutation(len(files2))[:num]
            dir_path1 = os.path.join(dst_path, dirs1[i] + dirs2[i])
            dir_path2 = os.path.join(dst_path, dirs2[i] + dirs1[i])
            if not os.path.exists(dir_path1):
                os.makedirs(dir_path1)
            if not os.path.exists(dir_path2):
                os.makedirs(dir_path2)
                name.append(os.path.split(dir_path1)[-1])
                name.append(os.path.split(dir_path2)[-1])
                for j in range(num):
                    img1 = cv2.imread(os.path.join(path, dirs1[i], files1[index1[j]]))
                    img2 = cv2.imread(os.path.join(path, dirs2[i], files2[index2[j]]))
                    img1 = cv2.resize(img1, (128, 256), interpolation=cv2.INTER_CUBIC)
                    img2 = cv2.resize(img2, (128, 256), interpolation=cv2.INTER_CUBIC)
                    img_new1 = np.concatenate(
                        (img1[:head, :, ], img2[head:, :, :]), 0)
                    img_new2 = np.concatenate(
                        (img2[:head, :, ], img1[head:, :, :]), 0)
                    c1 = files1[index1[j]].split('c')[1][0]
                    c2 = files2[index2[j]].split('c')[1][0]
                    file_name1 = dirs1[i] + dirs2[i] + '_' + str(j) + '_c' + c1 + '_' + files1[j].split('c')[1]
                    file_name2 = dirs2[i] + dirs1[i] + '_' + str(j) + '_c' + c2 + '_' + files2[j].split('c')[1]
                    cv2.imwrite(os.path.join(dir_path1, file_name1), img_new1)
                    cv2.imwrite(os.path.join(dir_path2, file_name2), img_new2)
            new_dir_num += 2
            new_file_num += num * 2
            debug_dirs = os.listdir(dst_path)
            if new_dir_num != len(debug_dirs):
                print('new_dir_num = %d   debug_dirs = %d' % (new_dir_num, len(debug_dirs)))
                print('dir_path1 = %s   dir_path2 = %s' % (dir_path1, dir_path2))
                exit()
            if new_dir_num >= max_id_num:
                break

    print('new_dir_num = %d  new_file_num = %d' % (new_dir_num, new_file_num))
    print('filter_dir num = %d' % len(dirs2))
    print('epoc = %d' % epoc)


def merge_dir():
    dirs = os.listdir(path)
    for dir in dirs:
        if not os.path.exists(os.path.join(dst_path, dir)):
            os.makedirs(os.path.join(dst_path, dir))
            files = os.listdir(os.path.join(path, dir))
            for file in files:
                shutil.copy(os.path.join(path, dir, file), os.path.join(dst_path, dir, file))
    dirs = os.listdir(dst_path)
    file_num = 0
    dir_num = len(dirs)
    for dir in dirs:
        file_num += len(os.listdir(os.path.join(dst_path, dir)))
    print('total dir_num = %d   file_num = %d' % (dir_num, file_num))

if __name__ == '__main__':
    if os.path.exists(dst_path):
        print('dst_path = %s is already existed and will be removed !!!' % dst_path)
        shutil.rmtree(dst_path)
    original_id_num = len(os.listdir(path))
    augment_id_num = 300
    print('opt.mode = %d   original_id_num = %d   augment_id_num = %d' % (opt.mode, original_id_num, augment_id_num))
    augment_more(augment_id_num)
    merge_dir()

