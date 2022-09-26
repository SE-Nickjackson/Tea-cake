import numpy as np
import torch
import pdb


class data_load:
    # 加载数据
    class Data:
        def __init__(self, data):
            self.x = data.astype(np.float32)
            self.y = data.astype(np.int64)

    def __init__(self, path):
        data, label = load_data_normalised(path)
        self.dt = self.Data(data)
        self.label = self.Data(label)


def get_data(path):
    #获取提取的特征与对应的标签
    data = np.load(path,allow_pickle=True)
    x_vector = data['vector']
    label_vector = data['utt']
    x_vector = np.array(x_vector)
    label_vector = np.array(label_vector)
    return x_vector, label_vector


def load_data_normalised(path):
    #获取X,与labels
    X, labels = get_data(path)
    X = X.astype(np.float32)
    return X, labels


def train_dataset_prepare(train_path, test_path):
    # training set voxceleb_4k_speaker
    print("loading training data from %s" % train_path);
    trn_data = data_load(train_path)
    print('.....')
    print(trn_data)
    t0_tensor = torch.from_numpy(trn_data.dt.x)
    print('.....')
    print(t0_tensor.shape)
    t0_label_tensor = torch.from_numpy(trn_data.label.y)
    print('.....')
    print(t0_label_tensor.shape)
    t0_dataset = torch.utils.data.TensorDataset(t0_tensor, t0_label_tensor)
    print('.........')
    print(t0_dataset)

    # testset: verify
    print("loading enrollment data from %s" % test_path);
    ver_data = data_load(test_path)
    t2_tensor = torch.from_numpy(ver_data.dt.x)
    t2_label_tensor = torch.from_numpy(ver_data.label.y)
    t2_dataset = torch.utils.data.TensorDataset(t2_tensor, t2_label_tensor)

    return t0_dataset,  t2_dataset



def verify_dataset_prepare(test_name=None):
    # testset: verify
    # print("loading enrollment data from %s" % test_name);
    ver_data = data_load(test_name)
    t2_tensor = torch.from_numpy(ver_data.dt.x)
    t2_label_tensor = torch.from_numpy(ver_data.label.y)
    t2_dataset = torch.utils.data.TensorDataset(t2_tensor, t2_label_tensor)
    return t2_dataset

