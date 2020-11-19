'''
create by xubing on 2020-11-11
计算企业之间的余弦相似度,并进行归一化。
'''
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
A = np.array([[1, 0], [1, 1], [1, 0]])  # 每一行一家企业


def calc_sim(ndarray):
    """
    利用企业的特征向量，计算企业间的相似度。
    ndarray: 各企业的特征向量
    return: 企业间的距离， 余弦相似度， 归一化后的相似度
    """
    dist = pairwise_distances(ndarray, metric='cosine')[0, :]
    cos_sim = 1 - dist
    norm_sim = np.around(cos_sim / cos_sim.sum(), decimals=3)

    return dist, cos_sim, norm_sim


print(calc_sim(A))
