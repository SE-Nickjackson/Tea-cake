import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from model.dataload import MyDataset


import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = models.resnet50(pretrained=True)

    def forward(self, input):
        output = self.net.conv1(input)
        output = self.net.bn1(output)
        output = self.net.relu(output)
        output = self.net.maxpool(output)
        output = self.net.layer1(output)
        output = self.net.layer2(output)
        output = self.net.layer3(output)
        output = self.net.layer4(output)
        output = self.net.avgpool(output)
        output = torch.flatten(output, 1)
        return output


model = torch.load("/home/daip/share/old_share/wxf/Tea_cake_CBIR/weights/subarc/ResNet50_bird_subarcloss_2022_7_18.pth")
print(model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
transforms = transforms.Compose([
    transforms.Resize((224,224)),  # 将图片短边缩放至256，长宽比保持不变：
    transforms.ToTensor()  # 把图片进行归一化，并把数据转换成Tensor类型
])
data_train = datasets.ImageFolder('/home/daip/share/old_share/wxf/datasets/bird_Species_24/train', transform=transforms)
data_test = datasets.ImageFolder('/home/daip/share/old_share/wxf/datasets/bird_Species_24/test',transform=transforms)
# 加载数据集
dataTrainLoader = torch.utils.data.DataLoader(data_train, batch_size=1, shuffle=True)
dataTestLoader = torch.utils.data.DataLoader(data_test, batch_size=1)
dog_test_data = []
dog_test_lable = []
for step, data in enumerate(dataTestLoader, start=0):
    images, labels = data
    labels = labels.to(device)
    outputs = model(images.to(device)).cpu()
    outputs = torch.squeeze(outputs)
    outputs = outputs.data.numpy()
    outputs = outputs.reshape(-1)
    labels = labels.cpu()
    labels = labels.data.numpy()
    labels = labels.reshape(-1)
    print(outputs.shape)
    print(labels)
    dog_test_data.append(outputs)
    dog_test_lable.append(labels)
    # lable.reshape(-1)
dog_lable = np.array(dog_test_lable)
dog_lable = dog_lable.reshape(-1)
print(len(dog_test_data))
np.savez('/home/daip/share/old_share/wxf/Tea_cake_CBIR/npz/subarc_bird.npz', vector=dog_test_data, utt=dog_lable)



