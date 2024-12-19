import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import random
import Model
import pickle
import matplotlib.pyplot as plt
import matplotlib
import untils

# 导入超参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device= ', device)
# 实验需求超参数
file_name = 'Ada_algorithm_QHN'
model_name, dataset_name, optimizer = 'GRU1', 'PTB', 'SGD'  # 模型、数据集与优化器
model_seed, data_seed = 42, 42  # 模型参数初始化与数据集分批所需随机种子
lr, Epoch, batch_size = 0.1, 80, 64  # 学习率、周期、样本分批大小
num_steps, token_type = 32, 'word'  # 一次输入的时间步数、词元类型
num_layer, embedding_dim, hidden_dim = 1, 128, 256  # GRU层数，嵌入层维度， 隐藏层单元数

# 按需求修改
# opt_list = ['Adabelief', 'Adabelief_QHN', 'YOGI', 'YOGI_QHN']  # 优化算法
# lr_list = [0.001, 0.002, 0.003, 0.005, 0.007, 0.01]  # 学习率
# # seed_list = torch.randint(10, 1000, (3, 2)).numpy()
# # print('seed_list:', seed_list)
# seed_list = [[825, 831], [253, 995], [192, 418], [629, 808], [390, 427]]  # 模型参数初始化与数据集分批所需随机种子
seed_list = [[825, 831], [253, 995], [192, 418]]  # 模型参数初始化与数据集分批所需随机种子
opt_list = ['Adabelief', 'Adabelief_QHN', 'YOGI', 'YOGI_QHN']  # 优化算法

L2 = ['t', 1e-4]      # L2正则    分别表示：是否启用、正则项系数
yr = ['t', 10, 0.1]   # 学习率预热 分别表示：是否启用、预热周期、初始学习率与固定学习率的比例
TH = ['t', 30, 0.2]     # 学习率退火 分别表示：是否启用、执行退火的间隔、退火系数

# 处理数据
if dataset_name == 'Alice':
    path = "../dataset/alice_in_wonderland.txt"
    # 获取数据和词表
    corpus, vocab = untils.load_corpus_vocab(path, token=token_type, max_tokens=-1)
    vocab_size = len(vocab)
    # 制作Alice数据集
    random.seed(42)  # 固定随机种子
    initial_indices = [random.randint(0, num_steps - 1) for _ in range(5)]  # 随机选择起始点，增强模型泛化
    x_data, y_data = [], []
    for i in range(len(initial_indices)):
        sub_corpus = corpus[initial_indices[i]:]
        num_subseqs = (len(sub_corpus) - 1) // num_steps
        for j in range(0, num_subseqs * num_steps, num_steps):
            x_data.append(sub_corpus[i: i + num_steps])
            y_data.append(sub_corpus[i + 1: i + 1 + num_steps])
    x_data = torch.tensor(x_data, dtype=torch.int64)
    y_data = torch.tensor(y_data, dtype=torch.int64)
    Alice_dataset = TensorDataset(x_data, y_data)
    # 划分训练数据和验证数据
    train_size = int(0.6 * len(Alice_dataset))
    val_size = len(Alice_dataset) - train_size
    train_dataset, val_dataset = random_split(Alice_dataset, [train_size, val_size])

elif dataset_name == 'PTB':
    train_path = "../dataset/PTB/ptb.train.txt"
    valid_path = "../dataset/PTB/ptb.valid.txt"
    train_data, vocab = untils.load_corpus_vocab(train_path, token=token_type, max_tokens=-1)
    vocab_size = len(vocab)
    # 获取验证数据
    valid_lines = untils.read_txt(valid_path)
    valid_tokens = untils.tokenize(valid_lines, token=token_type)
    val_data = [vocab[token] for line in valid_tokens for token in line]
    # 制作PTB的训练数据集和验证数据集
    x_train, y_train, x_val, y_val = [], [], [], []
    # # 训练数据
    num_subseqs = (len(train_data) - 1) // num_steps
    for i in range(0, num_subseqs * num_steps, num_steps):
        x_train.append(train_data[i: i + num_steps])
        y_train.append(train_data[i + 1: i + 1 + num_steps])
    x_train = torch.tensor(x_train, dtype=torch.int64)
    y_train = torch.tensor(y_train, dtype=torch.int64)
    # # 验证数据
    num_subseqs = (len(val_data) - 1) // num_steps
    for i in range(0, num_subseqs * num_steps, num_steps):
        x_val.append(val_data[i: i + num_steps])
        y_val.append(val_data[i+1: i+1 + num_steps])
    x_val = torch.tensor(x_val, dtype=torch.int64)
    y_val = torch.tensor(y_val, dtype=torch.int64)
    # # 数据集
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
else:
    print("dataset_name chu cuo!")

# 开始实验
for opt in range(len(opt_list)):
    optimizer = opt_list[opt]

    if optimizer == 'Adabelief':
        beta1, beta2 = 0.9, 0.999
        lr_list = [0.001, 0.002, 0.003, 0.005, 0.007, 0.01]
    elif optimizer == 'YOGI':
        beta1, beta2 = 0.9, 0.999
        lr_list = [0.005, 0.007, 0.01, 0.02, 0.03, 0.05]
    elif optimizer == 'Adabelief_QHN':
        beta1, beta2, v = 0.999, 0.999, 0.8
        lr_list = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1]
    elif optimizer == 'YOGI_QHN':
        beta1, beta2, v = 0.999, 0.999, 0.8
        lr_list = [0.005, 0.007, 0.01, 0.02, 0.03, 0.05]
    else:
        print('optimizer chu cuo!')

    for lr_index in range(len(lr_list)):
        lr = lr_list[lr_index]

        for seed_index in range(len(seed_list)):
            model_seed = seed_list[seed_index][0]
            data_seed = seed_list[seed_index][1]

            print('model:{}, dataset:{}, optimizer:{}'.format(model_name, dataset_name, optimizer))
            print('lr:{}, model_seed:{}, data_seed:{}'.format(lr, model_seed, data_seed))

            # 定义模型、损失函数
            if model_name == 'GRU1':
                model = Model.GRU1(vocab_size, embedding_dim, hidden_dim, num_layer, model_seed).to(device)
            else:
                print("model cuo wu!")
            criterion = nn.CrossEntropyLoss()
            torch.manual_seed(data_seed)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # 储存实验数据, 记录每一epoch后模型在训练集和验证集上的损失
            train_loss_list = []
            validation_loss_list = []

            # 优化器所需中间变量和超参数
            gx_lr = lr  # 实际参与更新的步长
            lamd = L2[1]  # L2正则系数
            eps = 1e-8
            fm = 1  # 用于计算beta1**t  修正动量项
            fh = 1  # 用于计算beta2**t  修正自适应步长项
            m = {}  # 记录动量项
            h = {}  # 记录自适应步长项
            lin_m = {}  # 记录修正后的动量项
            lin_h = {}  # 记录修正后的自适应步长项
            lin_w = {}  # 记录额外的模型参数(QHN的外推梯度)
            for name, param in model.named_parameters():
                m[name] = torch.zeros_like(param)
                h[name] = torch.zeros_like(param)
                lin_m[name] = torch.zeros_like(param)
                lin_h[name] = torch.zeros_like(param)
                lin_w[name] = param.data

            # 训练与测试模型
            for epoch in range(Epoch):

                # 学习率预热
                if yr[0] == 't':
                    if epoch + 1 <= yr[1]:
                        gx_lr = ((1 - yr[2]) * lr / (yr[1] - 1)) * epoch + yr[2] * lr
                # 学习率退火
                if TH[0] == 't':
                    if epoch % TH[1] == 0 and epoch != 0:
                        gx_lr = gx_lr * TH[2]

                # 训练模型
                model.train()  # 将模型设置为训练模式
                for i, (x, y) in enumerate(train_loader):

                    # 更新模型参数
                    if optimizer == 'Adabelief':
                        fm = fm * beta1
                        fh = fh * beta2
                        x, y = x.to(device), y.to(device)  # 将输入数据移动到GPU上
                        output = model(x)  # 前向传播
                        loss = criterion(output.view(-1, vocab_size), y.view(-1))  # 求损失
                        model.zero_grad()  # 将模型中所有参数的梯度清零
                        loss.backward()  # 反向传播求梯度
                        for name, param in model.named_parameters():
                            if L2[0] == 't':
                                param.grad = param.grad + lamd * param.data
                            # 更新动量项和自适应步长项
                            m[name] = (1 - beta1) * param.grad + beta1 * m[name]
                            h[name] = (1 - beta2) * ((param.grad - m[name]) ** 2) + beta2 * h[name]
                            # 修正动量项和自适应步长项
                            lin_m[name] = m[name] / (1 - fm)
                            lin_h[name] = eps + ((h[name] / (1 - fh)) ** 0.5)
                            # 更新模型参数
                            param.data = param.data - gx_lr * lin_m[name] / lin_h[name]

                    elif optimizer == 'YOGI':
                        fm = fm * beta1
                        fh = fh * beta2
                        x, y = x.to(device), y.to(device)  # 将输入数据移动到GPU上
                        output = model(x)  # 前向传播
                        loss = criterion(output.view(-1, vocab_size), y.view(-1))  # 求损失
                        model.zero_grad()  # 将模型中所有参数的梯度清零
                        loss.backward()  # 反向传播求梯度
                        for name, param in model.named_parameters():
                            if L2[0] == 't':
                                param.grad = param.grad + lamd * param.data
                            # 更新动量项和自适应步长项
                            m[name] = (1 - beta1) * param.grad + beta1 * m[name]
                            h[name] = h[name] - (1 - beta2) * torch.sign(h[name] - (param.grad ** 2)) * (
                                        param.grad ** 2)
                            # 修正动量项和自适应步长项
                            lin_m[name] = m[name] / (1 - fm)
                            lin_h[name] = eps + ((h[name] / (1 - fh)) ** 0.5)
                            # 更新模型参数
                            param.data = param.data - gx_lr * lin_m[name] / lin_h[name]

                    elif optimizer == 'Adabelief_QHN':
                        fm = fm * beta1
                        fh = fh * beta2
                        x, y = x.to(device), y.to(device)  # 将输入数据移动到GPU上
                        # 将模型参数外推用以计算外推梯度
                        for name, param in model.named_parameters():
                            param.data = param.data - gx_lr * (v / (1 - fm)) * beta1 * m[name]
                        output = model(x)  # 前向传播
                        loss = criterion(output.view(-1, vocab_size), y.view(-1))  # 求损失
                        model.zero_grad()  # 将模型中所有参数的梯度清零
                        loss.backward()  # 反向传播求梯度
                        for name, param in model.named_parameters():
                            if L2[0] == 't':
                                param.grad = param.grad + lamd * param.data
                            # 更新自适应步长项并修正当前外推梯度
                            h[name] = (1 - beta2) * ((beta1 * (param.grad - m[name])) ** 2) + beta2 * h[name]
                            param.grad = param.grad / (eps + ((h[name] / (1 - fh)) ** 0.5))
                            # 更新并修正动量项
                            m[name] = (1 - beta1) * param.grad + beta1 * m[name]
                            lin_m[name] = (v / (1 - fm)) * m[name] + (1 - v) * param.grad
                            # 更新模型参数并记录
                            param.data = lin_w[name] - gx_lr * lin_m[name]
                            lin_w[name] = param.data

                    elif optimizer == 'YOGI_QHN':
                        fm = fm * beta1
                        fh = fh * beta2
                        x, y = x.to(device), y.to(device)  # 将输入数据移动到GPU上
                        # 将模型参数外推用以计算外推梯度
                        for name, param in model.named_parameters():
                            param.data = param.data - gx_lr * (v / (1 - fm)) * beta1 * m[name]
                        output = model(x)  # 前向传播
                        loss = criterion(output.view(-1, vocab_size), y.view(-1))  # 求损失
                        model.zero_grad()  # 将模型中所有参数的梯度清零
                        loss.backward()  # 反向传播求梯度
                        for name, param in model.named_parameters():
                            if L2[0] == 't':
                                param.grad = param.grad + lamd * param.data
                            # 更新自适应步长项并修正当前外推梯度
                            h[name] = h[name] - (1 - beta2) * torch.sign(h[name] - (param.grad ** 2)) * (
                                        param.grad ** 2)
                            param.grad = param.grad / (eps + ((h[name] / (1 - fh)) ** 0.5))
                            # 更新并修正动量项
                            m[name] = (1 - beta1) * param.grad + beta1 * m[name]
                            lin_m[name] = (v / (1 - fm)) * m[name] + (1 - v) * param.grad
                            # 更新模型参数并记录
                            param.data = lin_w[name] - gx_lr * lin_m[name]
                            lin_w[name] = param.data
                    else:
                        print("geng xin chu cuo!")
                    if (i + 1) % 100 == 0:
                        print('Epoch: {}, iter: {}, loss: {:.6f}'
                              .format(epoch + 1, i + 1, float(loss.item())))

                # 模型设置为评估模式
                model.eval()
                with torch.no_grad():
                    # 评估训练集
                    total_loss = 0
                    for j, (x, y) in enumerate(train_loader):
                        # 将输入数据移动到GPU上
                        x, y = x.to(device), y.to(device)
                        output = model(x)
                        loss = criterion(output.view(-1, vocab_size), y.view(-1))  # 求损失
                        total_loss += loss

                    ave_loss = total_loss / len(train_loader)
                    train_loss_list.append(float(ave_loss.item()))

                    print('epoch{} loss{} on the train set'.format(epoch + 1, float(ave_loss.item())))

                    # 评估测试集
                    total_loss = 0
                    for k, (x, y) in enumerate(validation_loader):
                        # 将输入数据移动到GPU上
                        x, y = x.to(device), y.to(device)
                        output = model(x)
                        loss = criterion(output.view(-1, vocab_size), y.view(-1))  # 求损失
                        total_loss += loss

                    ave_loss = total_loss / len(validation_loader)
                    validation_loss_list.append(float(ave_loss.item()))

                    print('epoch{} loss{} on the test set'.format(epoch + 1, float(ave_loss.item())))

            # 保存实验数据
            file = '../save_data/{}_{}_{}/{}_lr{}_seed{}.pkl' \
                .format(file_name, model_name, dataset_name, optimizer, lr, seed_index)
            with open(file, 'wb') as f:
                pickle.dump({
                    'train_loss_list': train_loss_list,
                    'validation_loss_list': validation_loss_list,
                }, f)
            print("Train metrics saved successfully.")

        all_train_loss = np.zeros(Epoch)
        all_validation_loss = np.zeros(Epoch)

        L = len(seed_list)

        for p in range(L):
            file = '../save_data/{}_{}_{}/{}_lr{}_seed{}.pkl'\
                .format(file_name, model_name, dataset_name, optimizer, lr, p)
            with open(file, 'rb') as f:
                data = pickle.load(f)
            train_loss = np.array(data['train_loss_list'])
            validation_loss = np.array(data['validation_loss_list'])

            all_train_loss += train_loss
            all_validation_loss += validation_loss

        ave_train_loss = all_train_loss / L
        ave_validation_loss = all_validation_loss / L

        ave_train_loss = ave_train_loss.tolist()
        ave_validation_loss = ave_validation_loss.tolist()

        file = '../save_data/{}_{}_{}/ave_{}_lr{}.pkl'.format(file_name, model_name, dataset_name, optimizer, lr)
        with open(file, 'wb') as f:
            pickle.dump({
                'train_loss_list': ave_train_loss,
                'validation_loss_list': ave_validation_loss,
            }, f)
        print("save successfully")

    colors = matplotlib.colormaps.get_cmap('tab10')  # 颜色映射
    plt.figure(figsize=(10, 6))
    for index in range(len(lr_list)):
        file = '../save_data/{}_{}_{}/ave_{}_lr{}.pkl'\
            .format(file_name, model_name, dataset_name, optimizer, lr_list[index])
        with open(file, 'rb') as f:
            data = pickle.load(f)
        train_loss = np.array(data['train_loss_list'])
        validation_loss = np.array(data['validation_loss_list'])

        plt.plot(train_loss, label='{} lr{} train'
                 .format(optimizer, lr_list[index]), color=colors(index))
        plt.plot(validation_loss, label='{} lr{} validation'
                 .format(optimizer, lr_list[index]), linestyle='--', color=colors(index))
    plt.plot(np.full(Epoch, 4), label='4')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig('../save_tu/{}_{}_{}/{}_loss.png'.format(file_name, model_name, dataset_name, optimizer))
    plt.clf()  # 清除图像内容
    plt.close()  # 关闭图像

    print("tu saved successfully.")
