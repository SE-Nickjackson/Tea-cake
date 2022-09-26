from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import  DataLoader
# -----------------ready the dataset--------------------------
def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    # 构造函数带有默认参数
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            # 移除字符串首尾的换行符
            # 删除末尾空
            # 以空格为分隔符 将字符串分成
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            # print(words[0][39:])
            imgs.append(('/home/daip/share/old_share/wxf/datasets/tea-cake'+words[0][39:], int(words[1])))  # imgs中包含有图像路径和标签
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        # 调用定义的loader方法
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

