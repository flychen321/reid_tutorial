import numpy as np
import os
import cv2
import shutil
import argparse

# image size: 128 * 64 *3
parser = argparse.ArgumentParser(description='Augment')
parser.add_argument('--data_dir', default='new_dataset', type=str, help='data_dir')
parser.add_argument('--mode', default=5, type=int, help='mode')
opt = parser.parse_args()
print('opt = %s' % opt)
data_dir = opt.data_dir
print('data_dir = %s' % data_dir)
print('opt.mode = %s' % opt.mode)
src_path = 'miner_original'
merge_path = 'miner_merge'
dst_path = 'miner_new'
print('src_path = %s    dst_path = %s' % (src_path, dst_path))


def move_to_one_fold(src='miner_original', dst='miner_one'):
    dirs = os.listdir(src)
    if os.path.exists(dst):
        print('dst = %s is already existed and will be removed !!!' % dst)
        shutil.rmtree(dst)
    os.makedirs(dst)
    for dir in dirs:
        files = os.listdir(os.path.join(src, dir))
        for file in files:
            shutil.copy(os.path.join(src, dir, file), os.path.join(dst, file))


def augment_more(max_id_num=1000, ratio=0.2):
    dirs1 = os.listdir(src_path)
    dirs2 = os.listdir(src_path)
    dir_len = len(dirs2)
    name = []
    new_dir_num = 0
    new_file_num = 0
    epoc = 0
    while new_dir_num < max_id_num and epoc < 10:
        epoc += 1
        np.random.shuffle(dirs2)
        for i in range(int(len(dirs2))):
            dir_name = dirs1[i] + dirs2[i]
            files1 = os.listdir(os.path.join(src_path, dirs1[i]))
            files2 = os.listdir(os.path.join(src_path, dirs2[i]))
            num = min(len(files1), len(files2))
            if num < 6 or dir_name in name or dirs1[i] == dirs2[i]:
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
                    img1 = cv2.imread(os.path.join(src_path, dirs1[i], files1[index1[j]]))
                    img2 = cv2.imread(os.path.join(src_path, dirs2[i], files2[index2[j]]))
                    # img1 = cv2.resize(img1, (128, 256), interpolation=cv2.INTER_CUBIC)
                    # img2 = cv2.resize(img2, (128, 256), interpolation=cv2.INTER_CUBIC)
                    # img_new1 = np.concatenate(
                    #     (img1[:int(img1.shape[0] / 2), :, ], img2[int(img2.shape[0] / 2):, :, :]), 0)
                    # img_new2 = np.concatenate(
                    #     (img2[:int(img2.shape[0] / 2), :, ], img1[int(img1.shape[0] / 2):, :, :]), 0)

                    # target size is 256*128
                    height1 = int(img1.shape[0] * (128.0 / img1.shape[1]))
                    height2 = int(img2.shape[0] * (128.0 / img2.shape[1]))
                    img1 = cv2.resize(img1, (128, height1), interpolation=cv2.INTER_CUBIC)
                    img2 = cv2.resize(img2, (128, height2), interpolation=cv2.INTER_CUBIC)
                    img_new1 = np.concatenate(
                        (img1[:int(ratio * img1.shape[0]), :, ], img2[int(ratio * img2.shape[0]):, :, :]), 0)
                    img_new2 = np.concatenate(
                        (img2[:int(ratio * img2.shape[0]), :, ], img1[int(ratio * img1.shape[0]):, :, :]), 0)
                    file_name1 = dirs1[i] + dirs2[i] + '_' + str(j) + '_c' + files1[j].split('c')[1]
                    file_name2 = dirs2[i] + dirs1[i] + '_' + str(j) + '_c' + files2[j].split('c')[1]
                    cv2.imwrite(os.path.join(dir_path1, file_name1), img_new1)
                    cv2.imwrite(os.path.join(dir_path2, file_name2), img_new2)
            new_dir_num += 2
            new_file_num += num * 2
            dubug_dirs = os.listdir(dst_path)
            if new_dir_num != len(dubug_dirs):
                print('new_dir_num = %d   dubug_dirs = %d' % (new_dir_num, len(dubug_dirs)))
                print('dir_path1 = %s   dir_path2 = %s' % (dir_path1, dir_path2))
                exit()
            if new_dir_num >= max_id_num:
                break

    print('new_dir_num = %d  new_file_num = %d' % (new_dir_num, new_file_num))
    print('filter_dir num = %d' % len(dirs2))
    print('epoc = %d' % epoc)


def merge_dir():
    dirs = os.listdir(src_path)
    for dir in dirs:
        if not os.path.exists(os.path.join(merge_path, dir)):
            os.makedirs(os.path.join(merge_path, dir))
            files = os.listdir(os.path.join(src_path, dir))
            for file in files:
                shutil.copy(os.path.join(src_path, dir, file), os.path.join(merge_path, dir, file))
    dirs = os.listdir(merge_path)
    file_num = 0
    dir_num = len(dirs)
    for dir in dirs:
        file_num += len(os.listdir(os.path.join(merge_path, dir)))
    print('total dir_num = %d   file_num = %d' % (dir_num, file_num))


if __name__ == '__main__':
    move_to_one_fold(src='data/market1/pytorch/dataset_yunxiao', dst='data/market1/pytorch/miner_one')
    # if os.path.exists(dst_path):
    #     print('dst_path = %s is already existed and will be removed !!!' % dst_path)
    #     shutil.rmtree(dst_path)
    # original_id_num = len(os.listdir(src_path))
    # augment_id_num = int(100)
    # print('opt.mode = %d   original_id_num = %d   augment_id_num = %d' % (opt.mode, original_id_num, augment_id_num))
    # augment_more(augment_id_num)
    # # merge_dir()
