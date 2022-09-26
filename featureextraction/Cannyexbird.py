import os,glob
import cv2
from skimage.feature import local_binary_pattern
import _pickle as pickle
import scipy.spatial
import numpy as np
from sklearn.metrics import classification_report
# fh = open('/Volumes/wxf/TeaExperiment/TeaExperiment/data/teatest.txt','r')
data = []
lable = []
path = '/home/daip/share/old_share/wxf/Tea_cake_CBIR/npz/cannybird.npz'
radius = 1	# LBP算法中范围半径的取值
n_points = 8 * radius # 领域像素点数
def extractor(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # 求X方向上的梯度
    grad_x = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0)
    # 求y方向上的梯度
    grad_y = cv2.Sobel(gray, cv2.CV_16SC1, 0, 1)
    # 将梯度值转化到8位上来
    x_grad = cv2.convertScaleAbs(grad_x)
    y_grad = cv2.convertScaleAbs(grad_y)
    # 将两个梯度组合起来
    src1 = cv2.addWeighted(x_grad, 0.5, y_grad, 0.5, 0)
    # 组合梯度用canny算法，其中50和100为阈值
    edge = cv2.Canny(src1, 50, 100)
    # cv2.imshow("Canny_edge_1", edge)
    edge1 = cv2.Canny(grad_x, grad_y, 10, 100)
    # cv2.imshow("Canny_edge_2", edge1)
    # 用边缘做掩模，进行bitwise_and位运算
    edge2 = cv2.bitwise_and(image, image, mask=edge1)
    return edge2.reshape(-1)

file_dir = '/home/daip/share/old_share/wxf/datasets/bird_Species_24/test/'
class_names = os.listdir(file_dir)
for index,class_name in enumerate(class_names):
    for image_name in os.listdir(file_dir+class_name):
        # print(image_name)
        img = cv2.imread(file_dir+class_name+'/'+image_name)
        img = cv2.resize(img,(224,224))
        feature = extractor(img)
        print(feature)
        data.append(feature)
        lable.append(index)
        print(index)
print(len(lable))
print(len(data))
np.savez(path, vector=data, utt=lable)

# for line in fh:
#     # 移除字符串首尾的换行符
#     # 删除末尾空
#     # 以空格为分隔符 将字符串分成
#     line = line.strip('\n')
#     line = line.rstrip()
#     words = line.split()
#     img_path = '/Volumes/wxf/'+words[0][6:]
#     id = int(words[1])
#     img = cv2.imread(img_path)
#     feature = extractor(img)
#     data.append(feature)
#     lable.append(id)
#     print(id)
# np.savez(path, vector=data, utt=lable)

