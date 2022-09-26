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


model = torch.load("/home/daip/share/old_share/wxf/Tea_cake_CBIR/weights/triple/ResNet50_tea-cake_Tripletloss_2022_7_18_3.pth")
print(model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
root = "/home/daip/share/old_share/wxf/datasets/tea-cake/"
teadata = MyDataset(txt=root + 'teadata.txt', transform=transforms.ToTensor())
tea_data = DataLoader(teadata,batch_size=1,shuffle=True)
tea_all_data = []
tea_all_lable = []
for step, data in enumerate(tea_data, start=0):
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
    tea_all_data.append(outputs)
    tea_all_lable.append(labels)
    # lable.reshape(-1)
tea_lable = np.array(tea_all_lable)
tea_lable = tea_lable.reshape(-1)
print(len(tea_all_data))
np.savez('/home/daip/share/old_share/wxf/Tea_cake_CBIR/npz/tea_triple.npz', vector=tea_all_data, utt=tea_lable)



