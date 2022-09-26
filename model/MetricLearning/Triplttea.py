import torch
import torch.nn as nn
import os
import torch.optim as optim
from torch.utils.data import DataLoader

from model.dataload import MyDataset

### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ###
from torchvision import datasets, transforms, models

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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

### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ###
def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(
                "Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(
                    epoch, batch_idx, loss, mining_func.num_triplets
                )
            )


### convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def test(train_set, test_set, model, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, train_embeddings, test_labels, train_labels, False
    )
    print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transform = transforms.Compose(
#     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
# )





# trainDataPath = '/home/daip/share/old_share/wxh/ReID/dataset/image_classification2/train/'
# testDataPath = '/home/daip/share/old_share/wxh/ReID/dataset/image_classification2/test/'
# dataPath = '/home/daip/share/old_share/wxh/ReID/dataset/image_classification2/'

# transforms = transforms.Compose([
#     transforms.Resize((224,224)),  # 将图片短边缩放至256，长宽比保持不变：
#     transforms.ToTensor()  # 把图片进行归一化，并把数据转换成Tensor类型
# ])
root = "/home/daip/share/old_share/wxf/datasets/tea-cake/"
# data_train = datasets.ImageFolder('/home/daip/share/old_share/wxf/Tea_cake_CBIR/data', transform=transforms)
# traindata = MyDataset(txt=root + 'teatrain.txt', transform=transforms.ToTensor())
# train_data = DataLoader(data_train,batch_size=4,shuffle=True)
testdata = MyDataset(txt=root + 'teatest.txt', transform=transforms.ToTensor())
test_data = DataLoader(testdata,batch_size=64,shuffle=False)
# # 导入数据集
# data_train = datasets.ImageFolder(trainDataPath, transform=transforms)
# data_test = datasets.ImageFolder(testDataPath,transform=transforms)
# # 加载数据集
# dataTrainLoader = torch.utils.data.DataLoader(data_train, batch_size=64, shuffle=True)
# dataTestLoader = torch.utils.data.DataLoader(data_test, batch_size=64)

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 5


### pytorch-metric-learning stuff ###
distance = distances.CosineSimilarity()
reducer = reducers.ThresholdReducer(low=0)
loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
mining_func = miners.TripletMarginMiner(
    margin=0.2, distance=distance, type_of_triplets="semihard"
)
accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)
### pytorch-metric-learning stuff ###

# dataset1 = datasets.MNIST(".", train=True, download=True, transform=transform)
# dataset2 = datasets.MNIST(".", train=False, transform=transform)
# train_loader = torch.utils.data.DataLoader(dataset1, batch_size=256, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset2, batch_size=256)
for epoch in range(1, num_epochs + 1):
    train(model, loss_func, mining_func, device, test_data, optimizer, epoch)
    # test(traindata, testdata, model, accuracy_calculator)
torch.save(model, "/home/daip/share/old_share/wxf/Tea_cake_CBIR/weights/triple/ResNet50_tea-cake_Tripletloss_2022_7_18_3.pth")
print("模型已保存")