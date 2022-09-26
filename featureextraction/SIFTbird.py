import os
import cv2

import numpy as np

data = []
lable = []
path = '/home/daip/share/old_share/wxf/Tea_cake_CBIR/npz/SIFTbird.npz'
radius = 1	# LBP算法中范围半径的取值
n_points = 8 * radius # 领域像素点数
def extract_features(image_path,vector_size=32):
    '''
    :param image:传入读取的图片
    :param vector_size:超参数用来控制特征提取的个数
    :return:提取的特征
    '''
    try:
        image = cv2.imread(image_path)
        # image = cv2.resize(image, (224, 224))
        alg = cv2.xfeatures2d.SIFT_create()#SIFT特征提取器
        kps = alg.detect(image)#特征点
        kps = sorted(kps,key=lambda x:-x.response)[:vector_size]#前32个特征点
        kps,dsc = alg.compute(image,kps)
        dsc = dsc.flatten()
        needs_size = (vector_size*64)#特征算子
        if dsc.size < needs_size:
            dsc = np.concatenate([dsc,np.zeros(needs_size-dsc.size)])#提取2048个特征点
    except cv2.error as e:
        print('error:',e)
        return None
    return dsc
file_dir = '/home/daip/share/old_share/wxf/datasets/bird_Species_24/test/'
class_names = os.listdir(file_dir)
for index,class_name in enumerate(class_names):
    for image_name in os.listdir(file_dir+class_name):
        # print(image_name)


        feature = extract_features(file_dir+class_name+'/'+image_name)
        print(len(feature))
        data.append(feature)
        lable.append(index)
        print(index)
print(len(lable))
print(len(data))
np.savez(path, vector=data, utt=lable)

