import scipy.io
import torch
import numpy as np
import time
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse
import shutil
#######################################################################
# Evaluate

# image size: 128 * 64 *3
parser = argparse.ArgumentParser(description='Evaluate')
parser.add_argument('--mode', default=1, type=int, help='mode')
opt = parser.parse_args()
print('opt = %s' % opt)
print('opt.mode = %s' % opt.mode)

cam_metric = torch.zeros(15, 15)


def evaluate(qf, ql, qc, gf, gl, gc, qn=None, gn=None):
    query = qf.view(-1, 1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    # good index
    query_index = np.argwhere(gl == ql)
    # same camera
    camera_index = np.argwhere(gc == qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)  # .flatten())

    CMC_tmp = compute_mAP(index, qc, good_index, junk_index, qn, gn)
    return CMC_tmp


def compute_mAP(index, qc, good_index, junk_index, qn, gn):
    global dir_name
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    ranked_camera = gallery_cam[index]
    mask = np.in1d(index, junk_index, invert=True)
    # mask2 = np.in1d(index, np.append(good_index,junk_index), invert=True)
    index = index[mask]
    ranked_camera = ranked_camera[mask]
    for i in range(10):
        cam_metric[qc - 1, ranked_camera[i] - 1] += 1

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    dir_n = os.path.join('data/market/retrieval_result', qn.strip()[:4])
    if os.path.exists(dir_n):
        shutil.rmtree(dir_n)
    os.mkdir(dir_n)
    shutil.copy(os.path.join('data/market/query', qn.strip()),
                os.path.join(dir_n, '-'+qn.strip()))
    for i in np.arange(min(100, len(index))):
        name = str(i).zfill(2) + '_' + gn[index][i].strip()
        shutil.copy(os.path.join('data/market/bounding_box_test', gn[index][i].strip()),
                    os.path.join(dir_n, name))

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


######################################################################
result = scipy.io.loadmat('pytorch_result.mat')
query_feature = torch.FloatTensor(result['query_f'])
query_name = result['query_name']
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_name = result['gallery_name']
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

print(query_feature.shape)
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
right_cnt = 0
former_right_cnt = 0
former_i = 0
# print(query_label)

if os.path.exists('data/market/retrieval_result'):
    shutil.rmtree('data/market/retrieval_result')
os.mkdir('data/market/retrieval_result')

for i in range(len(query_label)):
    ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], query_cam[i], gallery_feature, gallery_label,
                               gallery_cam, query_name[i], gallery_name)
    if CMC_tmp[0] == -1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp

    if CMC_tmp[0].numpy() == 1:
        right_cnt += 1
    if i % 100 == 0 or i == len(query_label) - 1:
        print('i = %4d    CMC_tmp[0] = %s  real-time rank1 = %.4f  avg rank1 = %.4f' % (
        i, CMC_tmp[0].numpy(), float(right_cnt - former_right_cnt) / (i - former_i + 1), float(right_cnt) / (i + 1)))
        former_right_cnt = right_cnt
        former_i = i

CMC = CMC.float()
CMC = CMC / len(query_label)  # average CMC
print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))


