import os,glob
import cv2
from skimage.feature import local_binary_pattern
import _pickle as pickle
import scipy.spatial
import numpy as np
from sklearn.metrics import classification_report

data = []
lable = []
path = '/home/daip/share/old_share/wxf/Tea_cake_CBIR/npz/LBPbird.npz'
radius = 1	# LBP算法中范围半径的取值
n_points = 8 * radius # 领域像素点数
def extractor(image):

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(image, n_points, radius)
    lbp = lbp.reshape(-1)
    print(len(lbp))
    return lbp/255
file_dir = '/home/daip/share/old_share/wxf/datasets/bird_Species_24/test/'
class_names = os.listdir(file_dir)
for index,class_name in enumerate(class_names):
    for image_name in os.listdir(file_dir+class_name):
        # print(image_name)
        img = cv2.imread(file_dir+class_name+'/'+image_name)
        img = cv2.resize(img,(224,224))
        feature = extractor(img)
        # print(len(feature))
        data.append(feature)
        lable.append(index)
        print(index)
print(len(lable))
print(len(data))
np.savez(path, vector=data, utt=lable)

