import torch
import torch.nn as nn
import torch.nn.init as init


class PoolMLP(nn.Module):
    """用于 EMnist_digits"""
    def __init__(self, seed):
        super(PoolMLP, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

        self.seed = seed

        torch.manual_seed(self.seed)  # 设置随机种子以固定参数初始化
        nn.init.kaiming_normal_(self.fc1.weight, a=0.1)
        nn.init.kaiming_normal_(self.fc2.weight, a=0.1)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        x = nn.functional.leaky_relu(self.fc1(x), negative_slope=0.1)
        x = self.fc2(x)
        return x


class LSTM1(nn.Module):
    """用于 AG News"""
    def __init__(self, seed, vocab_size, embed_dim, hidden_size, num_classes):
        super(LSTM1, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.seed = seed

        torch.manual_seed(self.seed)  # 设置随机种子以固定参数初始化
        # 初始化embedding层
        init.xavier_uniform_(self.embedding.weight)  # 使用Xavier初始化embedding层的权重
        # 初始化LSTM层
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:  # 输入门权重
                init.xavier_uniform_(param)  # Xavier初始化
            elif 'weight_hh' in name:  # 隐藏门权重
                init.xavier_uniform_(param)  # Xavier初始化
            elif 'bias' in name:  # 偏置项
                init.zeros_(param)  # 偏置初始化为零
        # 初始化全连接层
        init.xavier_uniform_(self.fc.weight)  # 使用Xavier初始化全连接层的权重
        init.zeros_(self.fc.bias)  # 偏置初始化为零

    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embed_dim)

        # LSTM层
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out shape: (batch_size, seq_length, hidden_size)
        # hidden shape: (1, batch_size, hidden_size)

        # 取最后一个时间步的输出
        out = self.dropout(hidden.squeeze(0))
        out = self.fc(out)
        return out


class GRU1(nn.Module):
    """用于 AG News"""
    def __init__(self, seed, vocab_size, embed_dim, hidden_size, num_classes):
        super(GRU1, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.seed = seed

        torch.manual_seed(self.seed)  # 设置随机种子以固定参数初始化
        # 初始化embedding层
        init.xavier_uniform_(self.embedding.weight)  # 使用Xavier初始化embedding层的权重
        # 初始化GRU层
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:  # 输入门权重
                init.xavier_uniform_(param)  # Xavier初始化
            elif 'weight_hh' in name:  # 隐藏门权重
                init.xavier_uniform_(param)  # Xavier初始化
            elif 'bias' in name:  # 偏置项
                init.zeros_(param)  # 偏置初始化为零
        # 初始化全连接层
        init.xavier_uniform_(self.fc.weight)  # 使用Xavier初始化全连接层的权重
        init.zeros_(self.fc.bias)  # 偏置初始化为零

    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embed_dim)

        # GRU层
        gru_out, hidden = self.gru(embedded)
        # gru_out shape: (batch_size, seq_length, hidden_size)
        # hidden shape: (1, batch_size, hidden_size)

        # 取最后一个时间步的输出
        out = self.dropout(hidden.squeeze(0))
        out = self.fc(out)
        return out


class GLeNet5(nn.Module):
    """用于 FMnist """
    def __init__(self, seed):
        super(GLeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=0)
        self.fc1 = nn.Linear(32 * 5 * 5, 84)
        self.fc2 = nn.Linear(84, 10)
        self.seed = seed

        torch.manual_seed(self.seed)  # 设置随机种子以固定参数初始化
        nn.init.kaiming_normal_(self.conv1.weight, a=0.1)
        nn.init.kaiming_normal_(self.conv2.weight, a=0.1)
        nn.init.kaiming_normal_(self.fc1.weight, a=0.1)
        nn.init.kaiming_normal_(self.fc2.weight, a=0.1)

        nn.init.constant_(self.conv1.bias, 0.01)  # 使用常数初始化偏置
        nn.init.constant_(self.conv2.bias, 0.01)
        nn.init.constant_(self.fc1.bias, 0.01)
        nn.init.constant_(self.fc2.bias, 0.01)

    def forward(self, x):
        x = nn.functional.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.leaky_relu(self.conv2(x), negative_slope=0.1)
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 32 * 5 * 5)
        x = nn.functional.leaky_relu(self.fc1(x), negative_slope=0.1)
        x = self.fc2(x)
        return x


class LeNet5(nn.Module):
    """用于 EMnist_digits """
    def __init__(self, seed):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0)
        self.fc1 = nn.Linear(16 * 4 * 4, 84)
        self.fc2 = nn.Linear(84, 10)
        self.seed = seed

        torch.manual_seed(self.seed)  # 设置随机种子以固定参数初始化
        nn.init.kaiming_normal_(self.conv1.weight, a=0.1)
        nn.init.kaiming_normal_(self.conv2.weight, a=0.1)
        nn.init.kaiming_normal_(self.fc1.weight, a=0.1)
        nn.init.kaiming_normal_(self.fc2.weight, a=0.1)

        nn.init.constant_(self.conv1.bias, 0.01)  # 使用常数初始化偏置
        nn.init.constant_(self.conv2.bias, 0.01)
        nn.init.constant_(self.fc1.bias, 0.01)
        nn.init.constant_(self.fc2.bias, 0.01)

    def forward(self, x):
        x = nn.functional.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.leaky_relu(self.conv2(x), negative_slope=0.1)
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 16 * 4 * 4)
        x = nn.functional.leaky_relu(self.fc1(x), negative_slope=0.1)
        x = self.fc2(x)
        return x


class MobileCNN(nn.Module):
    """用于 EMnist_balanced """
    def __init__(self, seed):
        super(MobileCNN, self).__init__()
        self.features = nn.Sequential(
            # 常规卷积
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 深度可分离卷积
            nn.Conv2d(32, 32, 3, groups=16, padding=1),  # 深度卷积
            nn.Conv2d(32, 32, 1),                        # 逐点卷积
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 47)
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
        return self.classifier(x)


class TinyVGG(nn.Module):
    """用于 EMnist_letters """
    def __init__(self, seed):
        super(TinyVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # 28x28 → 28x28
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, 3, padding=0),  # 14x14 → 12x12
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 6 * 6, 128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(128, 26)  # 26类输出
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
        return self.classifier(x)


class MicroVGG(nn.Module):
    """用于 cifar10 """
    def __init__(self, seed):
        super(MicroVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
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


class VGG_cifar10(nn.Module):
    def __init__(self, seed):
        super(VGG_cifar10, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128), nn.LeakyReLU(negative_slope=0.1),
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


