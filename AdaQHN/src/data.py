import random
import torch
import pandas as pd
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
from collections import defaultdict


def get_subset(dataset, subset_size, seed):
    """均匀抽取子集（保持类别平衡）"""
    random.seed(seed)

    # 按类别分组索引
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    # 每类抽取 subset_size//n_class 个样本
    subset_indices = []
    n_class = len(class_indices)
    samples_per_class = subset_size // n_class
    for label in class_indices:
        subset_indices.extend(random.sample(class_indices[label], samples_per_class))

    return torch.utils.data.Subset(dataset, subset_indices)


def text_pipeline(text, max_length=64):
    """文本预处理：分词+截断/填充到固定长度"""
    tokens = text.lower().split()[:max_length]  # 截断
    if len(tokens) < max_length:
        tokens += ['<pad>'] * (max_length - len(tokens))  # 填充
    return tokens


def dataloader(seed, batch_size, dataset_name):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    num_workers = 2

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_FMnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(28, padding=2),  # 先四周填充0，再将图像随机裁剪成28*28
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    ])

    transform_EMnist = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x.transpose(1, 2)),  # 转置宽高
        # transforms.Lambda(lambda x: x.flip(2)),  # 水平翻转
        transforms.RandomCrop(28, padding=2),
    ])

    transform_cifar10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
    ])

    if dataset_name == 'FMnist':
        full_train = torchvision.datasets.FashionMNIST(root='../dataset', train=True, download=True,
                                                       transform=transform_FMnist)
        full_test = torchvision.datasets.FashionMNIST(root='../dataset', train=False, download=True,
                                                      transform=transform)
        train_dataset = get_subset(full_train, 6000, seed)
        test_dataset = get_subset(full_test, 1000, seed)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    elif dataset_name == 'EMnist_digits':
        full_train = torchvision.datasets.EMNIST(root='../dataset', split='digits', train=True, download=True,
                                                 transform=transform_EMnist)
        full_test = torchvision.datasets.EMNIST(root='../dataset', split='digits', train=False, download=True,
                                                transform=transform_EMnist)
        train_dataset = get_subset(full_train, 6000, seed)
        test_dataset = get_subset(full_test, 1000, seed)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    elif dataset_name == 'EMnist_letters':
        full_train = torchvision.datasets.EMNIST(root='../dataset', split='letters', train=True, download=True,
                                                 transform=transform_EMnist)
        full_test = torchvision.datasets.EMNIST(root='../dataset', split='letters', train=False, download=True,
                                                transform=transform_EMnist)
        full_train.targets -= 1  # 修正标签范围（原始标签1-26 → 改为0-25）
        full_test.targets -= 1
        train_dataset = get_subset(full_train, 6240, seed)
        test_dataset = get_subset(full_test, 1040, seed)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    elif dataset_name == 'EMnist_balanced':
        full_train = torchvision.datasets.EMNIST(root='../dataset', split='balanced', train=True, download=True,
                                                 transform=transform_EMnist)
        full_test = torchvision.datasets.EMNIST(root='../dataset', split='balanced', train=False, download=True,
                                                transform=transform_EMnist)
        train_dataset = get_subset(full_train, 11280, seed)
        test_dataset = get_subset(full_test, 1880, seed)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    elif dataset_name == 'cifar10':
        full_train = torchvision.datasets.CIFAR10(root='../dataset', train=True, download=True,
                                                  transform=transform_cifar10)
        full_test = torchvision.datasets.CIFAR10(root='../dataset', train=False, download=True,
                                                 transform=transform)
        train_dataset = get_subset(full_train, 5000, seed)
        test_dataset = get_subset(full_test, 1000, seed)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    elif dataset_name == 'AGnews':
        # 加载预处理好的数据
        train_df = pd.read_csv('../dataset/AG news/frac10_ml64_minfreq3_train_processed.csv')
        val_df = pd.read_csv('../dataset/AG news/frac10_ml64_minfreq3_val_processed.csv')

        train_indices = torch.tensor(train_df['Indices'].apply(eval).tolist(), dtype=torch.long)
        train_labels = torch.tensor(train_df['Class Index'].values, dtype=torch.long)

        val_indices = torch.tensor(val_df['Indices'].apply(eval).tolist(), dtype=torch.long)
        val_labels = torch.tensor(val_df['Class Index'].values, dtype=torch.long)

        # 创建TensorDataset
        train_dataset = torch.utils.data.TensorDataset(train_indices, train_labels)
        val_dataset = torch.utils.data.TensorDataset(val_indices, val_labels)

        # 创建DataLoader
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        print("dataset cuo wu")

    return train_loader, val_loader


