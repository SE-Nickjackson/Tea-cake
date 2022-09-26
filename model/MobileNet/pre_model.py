import torch.nn as nn
from torchvision import models


def model(name,num_class):
    if name == 'mobilenetv2':
        model = models.mobilenet_v2(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_class)
        return model
    if name == 'mobilenetv3_large':
        model = models.mobilenet_v3_large(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_class)
        return model
    if name == 'mobilenetv3_small':
        model = models.mobilenet_v3_small(pretrained=True)
        num_ftrs = model.classifier[1].in_features           
        model.classifier[1] = nn.Linear(num_ftrs, num_class) 
        return model

if __name__ == '__main__':
    model = model('mobilenetv3_large',24)
    print(model)