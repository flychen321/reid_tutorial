import os
import shutil
import glob
import numpy as np
# src_path = 'data/foot_original'
# mid_path = 'data/foot_mid'
# dst_path = 'data/foot'

src_path = 'data/gait_data'
mid_path = 'data/gait_mid'
dst_path = 'data/gait'

if os.path.exists(mid_path):
    shutil.rmtree(mid_path)
os.makedirs(mid_path)

dirs = os.listdir(src_path)
file_num = 0
# for dir in dirs:
#     mid_dir = os.path.join(mid_path, dir)
#     os.mkdir(mid_dir)
#     sub_dirs = os.listdir(os.path.join(src_path, dir))
#     for sub_dir in sub_dirs:
#         files = os.listdir(os.path.join(src_path, dir, sub_dir))
#         file_num += len(files)
#         for file in files:
#             shutil.copy(os.path.join(src_path, dir, sub_dir, file), os.path.join(mid_path, dir, file))
#     print(file_num)
#     exit()

dir_num = 0
for dir in dirs:
    mid_dir = os.path.join(mid_path, dir)
    os.mkdir(mid_dir)
    files = glob.glob(os.path.join(src_path, dir)+'/*/*.jpg')
    files += glob.glob(os.path.join(src_path, dir)+'/*.jpg')
    for file in files:
        shutil.copy(file, os.path.join(mid_path, dir, os.path.split(file)[-1]))
    file_num += len(files)
    dir_num += 1
    print(dir_num, file_num)

ratio_train = 0.8
ratio_query = 0.3
train_path = os.path.join(dst_path, 'pytorch', 'train_all')
query_path = os.path.join(dst_path, 'pytorch', 'query')
gallery_path = os.path.join(dst_path, 'pytorch', 'gallery')

if os.path.exists(dst_path):
    shutil.rmtree(dst_path)
os.makedirs(dst_path)
os.makedirs(train_path)
os.makedirs(query_path)
os.makedirs(gallery_path)
dirs = os.listdir(mid_path)
np.random.shuffle(dirs)
file_num = 0

dir_num = 0
for dir in dirs:
    files = glob.glob(os.path.join(mid_path, dir)+'/*.jpg')
    if dir_num < ratio_train * len(dirs):
        os.mkdir(os.path.join(train_path, dir))
        for file in files:
            shutil.copy(file, os.path.join(train_path, dir, os.path.split(file)[-1]))
    else:
        np.random.shuffle(files)
        file_cnt = 0
        os.mkdir(os.path.join(query_path, dir))
        os.mkdir(os.path.join(gallery_path, dir))
        for file in files:
            if file_cnt < ratio_query * len(files):
                shutil.copy(file, os.path.join(query_path, dir, os.path.split(file)[-1]))
            else:
                shutil.copy(file, os.path.join(gallery_path, dir, os.path.split(file)[-1]))
            file_cnt += 1


    file_num += len(files)
    dir_num += 1
    print(dir_num, file_num)