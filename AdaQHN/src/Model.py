import torch
import torch.nn as nn
import torch.nn.init as init


class GRU1(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, seed):
        super(GRU1, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seed = seed

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.5)  # Dropout层

        # 初始化各层权重
        nn.init.xavier_uniform_(self.embedding.weight)  # 使用 Xavier 均匀分布初始化

        torch.manual_seed(self.seed)  # 设置随机种子以固定参数初始化
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)  # 使用 Xavier 均匀分布初始化
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)  # 使用正交初始化
            elif 'bias' in name:
                # 初始化偏置
                nn.init.constant_(param, 0)  # 偏置初始化为0

        nn.init.xavier_uniform_(self.fc.weight)  # 使用 Xavier 均匀分布初始化
        nn.init.constant_(self.fc.bias, 0)  # 偏置初始化为0

    def forward(self, x):
        self.gru.flatten_parameters()  # 将权重压缩为一个连续的内存块，以避免不必要的性能损失
        x = self.embedding(x)  # [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        # 初始化隐藏状态h  形状保持不变一直为(num_layers, batch_size, hidden_size)
        h = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        # 经过GRU后out与h形状: (seq_len, batch_size, hidden_size),(num_layers, batch_size, hidden_size)
        out, h = self.gru(x, h)
        # Dropout一下，缓解过拟合
        out = self.dropout(out)
        # 经fc后out 变为 (seq_length, batch_size, vocab_size)
        out = self.fc(out)

        return out


class MLP1(nn.Module):
    def __init__(self, seed):
        super(MLP1, self).__init__()
        self.n = 64
        self.fc1 = nn.Linear(self.n, self.n)
        self.fc2 = nn.Linear(self.n, self.n)
        self.fc3 = nn.Linear(self.n, 1)
        self.seed = seed

        torch.manual_seed(self.seed)  # 设置随机种子以固定参数初始化
        nn.init.kaiming_normal_(self.fc1.weight, a=0.1)
        nn.init.kaiming_normal_(self.fc2.weight, a=0.1)
        nn.init.kaiming_normal_(self.fc3.weight, a=0.1)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, x):
        x = nn.functional.leaky_relu(self.fc1(x), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.fc2(x), negative_slope=0.1)
        x = torch.sigmoid(self.fc3(x))  # sigmoid用于将输出映射到01之间
        return x


class GLeNet(nn.Module):
    def __init__(self, seed):
        super(GLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.seed = seed

        torch.manual_seed(self.seed)  # 设置随机种子以固定参数初始化
        nn.init.kaiming_normal_(self.conv1.weight, a=0.1)
        nn.init.kaiming_normal_(self.conv2.weight, a=0.1)
        nn.init.kaiming_normal_(self.fc1.weight, a=0.1)
        nn.init.kaiming_normal_(self.fc2.weight, a=0.1)
        nn.init.kaiming_normal_(self.fc3.weight, a=0.1)

        nn.init.constant_(self.conv1.bias, 0.01)  # 使用常数初始化偏置
        nn.init.constant_(self.conv2.bias, 0.01)
        nn.init.constant_(self.fc1.bias, 0.01)
        nn.init.constant_(self.fc2.bias, 0.01)
        nn.init.constant_(self.fc3.bias, 0.01)

    def forward(self, x):
        x = nn.functional.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.leaky_relu(self.conv2(x), negative_slope=0.1)
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 32 * 5 * 5)
        x = nn.functional.leaky_relu(self.fc1(x), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.fc2(x), negative_slope=0.1)
        x = self.fc3(x)
        return x


class LeNet(nn.Module):
    def __init__(self, seed):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.seed = seed

        torch.manual_seed(self.seed)  # 设置随机种子以固定参数初始化
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)

        nn.init.constant_(self.conv1.bias, 0.01)  # 使用常数初始化偏置
        nn.init.constant_(self.conv2.bias, 0.01)
        nn.init.constant_(self.fc1.bias, 0.01)
        nn.init.constant_(self.fc2.bias, 0.01)
        nn.init.constant_(self.fc3.bias, 0.01)

    def forward(self, x):
        x = nn.functional.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.leaky_relu(self.conv2(x), negative_slope=0.1)
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 16 * 5 * 5)
        x = nn.functional.leaky_relu(self.fc1(x), negative_slope=0.1)
        x = nn.functional.leaky_relu(self.fc2(x), negative_slope=0.1)
        x = self.fc3(x)
        return x


class VGG_CIFAR10(nn.Module):
    def __init__(self, seed):
        super(VGG_CIFAR10, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 128), nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(128, 10)
        )
        self.seed = seed

        # 设置固定的随机种子
        torch.manual_seed(self.seed)

        # 手动初始化权重和偏置
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                init.xavier_uniform_(layer.weight)
                init.constant_(layer.bias, 0.01)

        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.constant_(layer.bias, 0.01)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG11(nn.Module):
    def __init__(self, seed):
        super(VGG11, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.LeakyReLU(negative_slope=0.1), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.LeakyReLU(negative_slope=0.1), nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class residual(nn.Module):
    def __init__(self, input_channels, output_channels, use_1conv=False, strides=1, seed=42):
        super(residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, stride=1)
        if use_1conv:
            self.conv3 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.seed = seed

        torch.manual_seed(self.seed)  # 设置随机种子以固定参数初始化
        nn.init.kaiming_normal_(self.conv1.weight, a=0.1)
        nn.init.kaiming_normal_(self.conv2.weight, a=0.1)
        nn.init.constant_(self.conv1.bias, 0.01)  # 使用常数初始化偏置
        nn.init.constant_(self.conv2.bias, 0.01)
        if use_1conv:
            nn.init.kaiming_normal_(self.conv3.weight)
            nn.init.constant_(self.conv3.bias, 0.01)

    def forward(self, x):
        y = nn.functional.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1)
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        y = nn.functional.leaky_relu(y, negative_slope=0.1)
        return y


# 18层的resnet
class ResNet_cifar10(nn.Module):
    def __init__(self, seed):
        super(ResNet_cifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(64)

        self.resnet_block1 = nn.Sequential(
            residual(64, 64, use_1conv=False, seed=seed),
            residual(64, 64, use_1conv=False, seed=seed)
        )
        self.resnet_block2 = nn.Sequential(
            residual(64, 128, use_1conv=True, strides=2, seed=seed),
            residual(128, 128, use_1conv=False, seed=seed)
        )
        self.resnet_block3 = nn.Sequential(
            residual(128, 256, use_1conv=True, strides=2, seed=seed),
            residual(256, 256, use_1conv=False, seed=seed)
        )
        self.resnet_block4 = nn.Sequential(
            residual(256, 512, use_1conv=True, strides=2, seed=seed),
            residual(512, 512, use_1conv=False, seed=seed)
        )

        self.AvgPool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, 10)

        torch.manual_seed(seed)  # 设置随机种子以固定参数初始化
        nn.init.kaiming_normal_(self.conv1.weight, a=0.1)
        nn.init.kaiming_normal_(self.fc.weight, a=0.1)
        nn.init.constant_(self.conv1.bias, 0.01)  # 使用常数初始化偏置
        nn.init.constant_(self.fc.bias, 0.01)

    def forward(self, x):
        x = nn.functional.leaky_relu(self.bn(self.conv1(x)), negative_slope=0.1)
        x = self.resnet_block1(x)
        x = self.resnet_block2(x)
        x = self.resnet_block3(x)
        x = self.resnet_block4(x)
        x = self.AvgPool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


