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
    # query = random.sample(all_feature_lable, int(len(all_feature_lable)*0.2*(1/3)))
    query = random.sample(all_feature_lable, int(len(all_feature_lable) * 0.2 ))
    query = np.array(query)
    query_feature = []
    for q in query[:,0]:
        query_feature.append(q)
    query_feature = np.array(query_feature)
    #生成Gallery
    gallery = random.sample(all_feature_lable, int(len(all_feature_lable)*0.8))
    gallery = np.array(gallery)

    gallery_feature = []
    for g in gallery[:,0]:
        gallery_feature.append(g)
    gallery_feature = np.array(gallery_feature)
    #计算欧式距离
    distance = cos_cdist(query_feature,gallery_feature)
    ap0 = []
    ap1 = []
    ap2 = []
    ap3 = []
    ap4 = []
    ap5 = []
    ap6 = []
    ap7 = []
    ap8 = []
    ap9 = []
    ap10 = []
    ap11 = []
    ap12 = []
    ap13 = []
    ap14 = []
    ap15 = []
    ap16 = []
    ap17 = []
    ap18 = []
    ap19 = []
    ap20 = []
    ap21 = []
    ap22 = []
    ap23 = []
    for i in range(0,len(query)):
        lable = query[i][1]
        id = np.argsort(distance[i])[:1]
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
        if lable == 4:
            ap4.append(sum)
        if lable == 5:
            ap5.append(sum)
        if lable == 6:
            ap6.append(sum)
        if lable == 7:
            ap7.append(sum)
        if lable == 8:
            ap8.append(sum)
        if lable == 9:
            ap9.append(sum)
        if lable == 10:
            ap10.append(sum)
        if lable == 11:
            ap11.append(sum)
        if lable == 12:
            ap12.append(sum)
        if lable == 13:
            ap13.append(sum)
        if lable == 14:
            ap14.append(sum)
        if lable == 15:
            ap15.append(sum)
        if lable == 16:
            ap16.append(sum)
        if lable == 17:
            ap17.append(sum)
        if lable == 18:
            ap18.append(sum)
        if lable == 19:
            ap19.append(sum)
        if lable == 20:
            ap20.append(sum)
        if lable == 21:
            ap21.append(sum)
        if lable == 22:
            ap22.append(sum)
        if lable == 23:
            ap23.append(sum)
    sumap = 0
    if ap0 != []:
        sumap = sumap+np.mean(ap0)
    if ap1 != []:
        sumap = sumap+np.mean(ap1)
    if ap2 != []:
        sumap = sumap+np.mean(ap2)
    if ap3 != []:
        sumap = sumap+np.mean(ap3)
    if ap4 != []:
        sumap = sumap+np.mean(ap4)
    if ap5 != []:
        sumap = sumap+np.mean(ap5)
    if ap6 != []:
        sumap = sumap+np.mean(ap6)
    if ap7 != []:
        sumap = sumap+np.mean(ap7)
    if ap8 != []:
        sumap = sumap+np.mean(ap8)
    if ap9 != []:
        sumap = sumap+np.mean(ap9)
    if ap10 != []:
        sumap = sumap+np.mean(ap10)
    if ap11 != []:
        sumap = sumap+np.mean(ap11)
    if ap12 != []:
        sumap = sumap+np.mean(ap12)
    if ap13 != []:
        sumap = sumap+np.mean(ap13)
    if ap14 != []:
        sumap = sumap+np.mean(ap14)
    if ap15 != []:
        sumap = sumap+np.mean(ap1)
    if ap16 != []:
        sumap = sumap+np.mean(ap16)
    if ap17 != []:
        sumap = sumap+np.mean(ap17)
    if ap18 != []:
        sumap = sumap+np.mean(ap18)
    if ap19 != []:
        sumap = sumap+np.mean(ap19)
    if ap20 != []:
        sumap = sumap+np.mean(ap20)
    if ap21 != []:
        sumap = sumap+np.mean(ap21)
    if ap22 != []:
        sumap = sumap+np.mean(ap22)
    if ap23 != []:
        sumap = sumap+np.mean(ap23)
    # map = (np.mean(ap0)+np.mean(ap1)+np.mean(ap2)+np.mean(ap3)+np.mean(ap4)+
    #        np.mean(ap5)+np.mean(ap6)+np.mean(ap7)+np.mean(ap8)+np.mean(ap9)+
    #        np.mean(ap10)+np.mean(ap11)+np.mean(ap12)+np.mean(ap13)+np.mean(ap14)+np.mean(ap15)+
    #        np.mean(ap16)+np.mean(ap17)+np.mean(ap18)+np.mean(ap19)+
    #        np.mean(ap20)+np.mean(ap21)+np.mean(ap22)+np.mean(ap23))/24
    map = sumap /24
    print('%.4f' % map)
for i in range(0,10):
    # main('/home/daip/share/old_share/wxf/Tea_cake_CBIR/DNF_model/z_trn_2022_7_20/z0_epoch34.npz')
    main('/home/daip/share/old_share/wxf/Tea_cake_CBIR/npz/subarc_bird.npz')





