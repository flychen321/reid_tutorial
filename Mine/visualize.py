import scipy.io
import torch
import numpy as np
import time
import os
import matplotlib
import argparse
import shutil
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE


# matplotlib.use('agg')


# 加载数据
def get_data():
    """
    :return: 数据集、标签、样本数量、特征数量
    """
    digits = datasets.load_digits(n_class=10)
    data = digits.data  # 图片特征
    label = digits.target  # 图片标签
    n_samples, n_features = data.shape  # 数据集的形状
    return data, label, n_samples, n_features


# 对样本进行预处理并画图
def plot_embedding(data, label, title):
    """
    :param data:数据集
    :param label:样本标签
    :param title:图像标题
    :return:图像
    """
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)  # 对数据进行归一化处理
    fig = plt.figure()  # 创建图形实例
    ax = plt.subplot(111)  # 创建子图
    # 遍历所有样本
    for i in range(data.shape[0]):
        # 在图中为每个数据点画出标签
        plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / 10),
                 fontdict={'weight': 'bold', 'size': 7})
    plt.xticks()  # 指定坐标的刻度
    plt.yticks()
    plt.title(title, fontsize=14)
    # 返回值
    return fig


# 主函数，执行t-SNE降维
def t_sne(data, label):
    # data, label , n_samples, n_features = get_data()		# 调用函数，获取数据集信息
    print('Starting compute t-SNE Embedding...')
    ts = TSNE(n_components=2, init='pca', random_state=0)
    # t-SNE降维
    reslut = ts.fit_transform(data)
    # 调用函数，绘制图像
    # fig = plot_embedding(reslut, label, 't-SNE Embedding of miners')
    title = 't-SNE Embedding of miners'
    plt.title(title, fontsize=14)
    for label1 in set(label):
        index = np.where(label1 == label)
        x = reslut[index][:, 0]
        y = reslut[index][:, 1]
        plt.scatter(x, y)
    # 显示图像
    plt.savefig('t-SNE.png')
    plt.show()


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

    dir_n = os.path.join('data/market1/pytorch/retrieval_result', qn.strip()[:4])
    if os.path.exists(dir_n):
        shutil.rmtree(dir_n)
    os.mkdir(dir_n)
    shutil.copy(os.path.join('data/market1/pytorch/miner_one', qn.strip()),
                os.path.join(dir_n, '-' + qn.strip()))
    for i in np.arange(min(100, len(index))):
        name = str(i).zfill(2) + '_' + gn[index][i].strip()
        if os.path.exists(os.path.join('data/market1/pytorch/miner_one', gn[index][i].strip())):
            shutil.copy(os.path.join('data/market1/pytorch/miner_one', gn[index][i].strip()), os.path.join(dir_n, name))
        elif os.path.exists(os.path.join('data/market1/pytorch/bounding_box_test', gn[index][i].strip())):
            shutil.copy(os.path.join('data/market1/pytorch/bounding_box_test', gn[index][i].strip()), os.path.join(dir_n, name))
        else:
            print("file=%s not exist!!!" % gn[index][i].strip())

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

# if len(query_label) < 200:
#     t_sne(query_feature, query_label)

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

print(query_feature.shape)
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
right_cnt = 0
former_right_cnt = 0
former_i = 0
# print(query_label)

if os.path.exists('data/market1/pytorch/retrieval_result'):
    shutil.rmtree('data/market1/pytorch/retrieval_result')
os.mkdir('data/market1/pytorch/retrieval_result')

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
            i, CMC_tmp[0].numpy(), float(right_cnt - former_right_cnt) / (i - former_i + 1),
            float(right_cnt) / (i + 1)))
        former_right_cnt = right_cnt
        former_i = i

CMC = CMC.float()
CMC = CMC / len(query_label)  # average CMC
print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))
