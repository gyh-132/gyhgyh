import random
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, random_split


def dataloader(seed, batch_size, dataset_name):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_cifar10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，再将图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    ])

    transform_FMnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(28, padding=1),  # 先四周填充0，再将图像随机裁剪成28*28
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    ])

    if dataset_name == 'FMnist':
        train_dataset = torchvision.datasets.FashionMNIST(root='../dataset', train=True, download=True,
                                                          transform=transform_FMnist)
        test_dataset = torchvision.datasets.FashionMNIST(root='../dataset', train=False, download=True,
                                                         transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    elif dataset_name == 'Mnist':
        train_dataset = torchvision.datasets.MNIST(root='../dataset', train=True, download=True,
                                                   transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='../dataset', train=False, download=True,
                                                  transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    elif dataset_name == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(root='../dataset', train=True, download=True,
                                                          transform=transform_cifar10)
        test_dataset = torchvision.datasets.CIFAR10(root='../dataset', train=False, download=True,
                                                         transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    elif dataset_name == 'DXS1':
        n = 64  # 数据维度
        # 初始化所需数据
        A = np.round(np.random.uniform(-1, 1, (n, n)), 2)  # 仅保留两位小数
        B = np.round(np.random.uniform(-1, 1, (n, 1)), 2)
        c = np.round(np.random.uniform(-1, 1), 2)
        X = np.round(np.random.uniform(-1, 1, (25600, n, 1)), 2)
        Y = np.zeros((25600, 1))
        # 计算y = (x^T)Ax + (B^T)x + c, x \in X
        for i in range(X.shape[0]):
            x = X[i]  # 获取第 i 个 x
            Y[i] = np.dot(np.dot(x.T, A), x) + np.dot(B.T, x) + c

        # 将 X 和 Y 转换为 Tensor
        X = X.reshape(25600, n)
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)
        # 对 Y 进行最大最小标准化处理
        min_Y, max_Y = Y.min(), Y.max()
        Y = (Y - min_Y) / (max_Y - min_Y)
        # 创建 TensorDataset
        dataset = TensorDataset(X, Y)
        # 划分训练集和验证集
        train_size = int(0.6 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        # 创建 DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    else:
        print("dataset cuo wu")

    return train_loader, val_loader

