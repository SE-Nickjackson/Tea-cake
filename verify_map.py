# -*- coding: utf-8 -*-
# @Time    : 2022/7/17 10:51
# @Author  : wxf
# @FileName: verify_map.py
# @Software: PyCharm
# @Email ：15735952634@163.com
import scipy.spatial
from DNF_model.dnfdataload import verify_dataset_prepare
import numpy as np
from sklearn.metrics import classification_report,average_precision_score
from maptest import get_mAP
from random import choice
#计算欧式距离
def cos_cdist(vector1,vector2):
    return scipy.spatial.distance.cdist(vector1, vector2, 'euclidean')#欧式距离
#获取数据
def main(path):
    t2_dataset = verify_dataset_prepare(path)
    #获取feature与lable
    data = t2_dataset[:][0].data.numpy()
    lable = t2_dataset[:][1].data.numpy()
    all_feature_lable = []
    for i in range(0,len(lable)):
        feature_lable = [data[i],lable[i]]
        all_feature_lable.append(feature_lable)
    # print(type(all_feature_lable))
    #生成query
    import random
    query = random.sample(all_feature_lable, int(len(all_feature_lable)*0.2*(1/3)))
    query = np.array(query)
    query_feature = []
    for q in query[:,0]:
        query_feature.append(q)
    query_feature = np.array(query_feature)
    #生成Gallery
    gallery = random.sample(all_feature_lable, int(len(all_feature_lable)*0.8*(1/3)))
    gallery = np.array(gallery)

    gallery_feature = []
    for g in gallery[:,0]:
        gallery_feature.append(g)
    gallery_feature = np.array(gallery_feature)
    #计算欧式距离
    distance = cos_cdist(query_feature,gallery_feature)
    # print(len(distance[0]))
    ap0 = []
    ap1 = []
    ap2 = []
    ap3 = []
    for i in range(0,len(query)):
        lable = query[i][1]
        id = np.argsort(distance[i])[:50]
        pre = []
        for id_i in id:
            pre.append(gallery[id_i][1])
        # print(pre)
        r = pre.count(lable)
        one_precsion = []
        num = 0
        for i in range(0,len(pre)):
            if pre[i] == lable:
                num = num+1
                one_precsion.append(num/(i+1))
        sum = 0
        for j in range (0,len(one_precsion)):
            if j == 0:
                sum = sum + ((one_precsion[j]+one_precsion[j])/2)*(1/r)
            else:
                sum = sum + ((one_precsion[j]+one_precsion[j-1])/2)*(1/r)
        if lable == 0:
            ap0.append(sum)
        if lable == 1:
            ap1.append(sum)
        if lable == 2:
            ap2.append(sum)
        if lable == 3:
            ap3.append(sum)
    map = (np.mean(ap0)+np.mean(ap1)+np.mean(ap2)+np.mean(ap3))/4
    print(map)
for i in range(0,10):
    main('/home/daip/share/old_share/wxf/Tea_cake_CBIR/npz/tea_triple.npz')