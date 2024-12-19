# SGD, momentum, NAG, RNAG, QHM, QHN
# QHM可恢复SGD(beta=0, v=1)、moment(beta不等于0, v=1)、R_NAG(beta任意, v=beta)
# QHN可恢复SGD(beta=0, v=1)、NAG(beta不等于0, v = 1)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import random
import Model
import pickle
import matplotlib.pyplot as plt
import untils

# 导入超参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device= ', device)
# 实验需求超参数
file_name = 'QHN'
model_name, dataset_name, optimizer = 'GRU1', 'PTB', 'SGD'  # 模型、数据集与优化器
model_seed, data_seed = 42, 42  # 模型参数初始化与数据集分批所需随机种子
lr, Epoch, batch_size = 0.1, 150, 64  # 学习率、周期、样本分批大小
num_steps, token_type = 32, 'word'  # 一次输入的时间步数、词元类型
num_layer, embedding_dim, hidden_dim = 1, 128, 256  # GRU层数，嵌入层维度， 隐藏层单元数

# 按需求修改
# seed_list = torch.randint(10, 1000, (3, 2)).numpy()
# print('seed_list:', seed_list)

# seed_list = [[825, 831], [253, 995], [192, 418], [629, 808], [390, 427]]
# bs_list = [64, 128, 256]
# lr_list = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  # SGD
# lr_list = [0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1]  # 其它
# beta_list = [0.8, 0.9, 0.95, 0.99, 0.995]  # Momentum、NAG、RNAG
# beta_list = [0.9, 0.95, 0.99, 0.995, 0.999, 0.9995]  # QHM、QHN
# v_list = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
# opt_list = ['SGD', 'momentum', 'NAG', 'RNAG', 'QHM', 'QHN']
lr_list = [1]
seed_list = [[825, 831], [253, 995], [192, 418]]
opt_list = ['SGD', 'momentum', 'NAG', 'RNAG', 'QHM', 'QHN']

L2 = ['t', 1e-4]      # L2正则    分别表示：是否启用、正则项系数
yr = ['t', 10, 0.1]   # 学习率预热 分别表示：是否启用、预热周期、初始学习率与固定学习率的比例
TH = ['t', 100, 0.1]     # 学习率退火 分别表示：是否启用、执行退火的间隔、退火系数

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
for lr_index in range(len(lr_list)):
    lr = lr_list[lr_index]

    for opt in range(len(opt_list)):
        optimizer = opt_list[opt]

        if optimizer == 'SGD':
            beta_list = [0.9]
            v_list = [1]
        elif optimizer == 'momentum':
            beta_list = [0.9]
            v_list = [1]
        elif optimizer == 'NAG':
            beta_list = [0.9]
            v_list = [1]
        elif optimizer == 'RNAG':
            beta_list = [0.9]
            v_list = [1]
        elif optimizer == 'QHM':
            beta_list = [0.999]
            v_list = [0.7]
        elif optimizer == 'QHN':
            beta_list = [0.999]
            v_list = [0.8]
        else:
            print('optimizer chu cuo!')

        for beta_index in range(len(beta_list)):
            beta = beta_list[beta_index]

            for v_index in range(len(v_list)):
                v = v_list[v_index]

                for seed_index in range(len(seed_list)):
                    model_seed = seed_list[seed_index][0]
                    data_seed = seed_list[seed_index][1]

                    print('model:{}, dataset:{}, model_seed:{}, data_seed:{}' \
                          .format(model_name, dataset_name, model_seed, data_seed))
                    print('lr:{}, Epoch:{}, batch_size:{}'.format(lr, Epoch, batch_size))
                    print('optimizer:{}, beta:{}, v:{}'.format(optimizer, beta, v))
                    print('L2:{}, yr:{}, TH:{}'.format(L2, yr, TH))

                    # 定义模型、损失函数和数据迭代器
                    if model_name == 'GRU1':
                        model = Model.GRU1(vocab_size, embedding_dim, hidden_dim, num_layer, model_seed).to(device)
                    else:
                        print("model cuo wu!")
                    criterion = nn.CrossEntropyLoss()
                    torch.manual_seed(data_seed)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                    # 储存实验数据
                    # 记录每一epoch后模型在训练集和验证集上的损失
                    train_loss_list = []
                    validation_loss_list = []

                    # 优化器所需中间变量和超参数
                    lamd = L2[1]  # 正则系数
                    fm = 1  # 用于计算beta**t
                    m = {}  # 记录动量项
                    lin_w = {}  # 记录模型参数中间变量(NAG专用)
                    for name, param in model.named_parameters():
                        lin_w[name] = param.data
                        m[name] = torch.zeros_like(param)

                    gx_lr = lr

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

                            # 更新模型参数(QHM)
                            if optimizer == 'QHM':
                                fm = fm * beta
                                x, y = x.to(device), y.to(device)  # 将输入数据移动到GPU上
                                output = model(x)  # 前向传播
                                loss = criterion(output.view(-1, vocab_size), y.view(-1))  # 求损失
                                # loss = torch.exp(criterion(output.view(-1, vocab_size), y.view(-1)))  # 求损失
                                model.zero_grad()  # 将模型中所有参数的梯度清零
                                loss.backward()  # 反向传播求梯度
                                for name, param in model.named_parameters():
                                    if L2[0] == 't':
                                        param.grad = param.grad + lamd * param.data
                                    m[name] = (1 - beta) * param.grad + beta * m[name]
                                    param.data = param.data - gx_lr * ((1 - v) * param.grad + (v / (1 - fm)) * m[name])

                            # 更新模型参数(QHN)
                            elif optimizer == 'QHN':
                                fm = fm * beta
                                x, y = x.to(device), y.to(device)  # 将输入数据移动到GPU上
                                for name, param in model.named_parameters():
                                    param.data = param.data - gx_lr * (v / (1 - fm)) * beta * m[name]
                                output = model(x)  # 前向传播
                                loss = criterion(output.view(-1, vocab_size), y.view(-1))  # 求损失
                                # loss = torch.exp(criterion(output.view(-1, vocab_size), y.view(-1)))  # 求损失
                                model.zero_grad()  # 将模型中所有参数的梯度清零
                                loss.backward()  # 反向传播求梯度
                                for name, param in model.named_parameters():
                                    if L2[0] == 't':
                                        param.grad = param.grad + lamd * param.data
                                    m[name] = (1 - beta) * param.grad + beta * m[name]
                                    param.data = lin_w[name] - gx_lr * ((1 - v) * param.grad + (v / (1 - fm)) * m[name])
                                    # 记录当前模型参数
                                    lin_w[name] = param.data

                            elif optimizer == 'SGD':
                                x, y = x.to(device), y.to(device)  # 将输入数据移动到GPU上
                                output = model(x)  # 前向传播
                                loss = criterion(output.view(-1, vocab_size), y.view(-1))  # 求损失
                                # loss = torch.exp(criterion(output.view(-1, vocab_size), y.view(-1)))  # 求损失
                                model.zero_grad()  # 将模型中所有参数的梯度清零
                                loss.backward()  # 反向传播求梯度
                                for name, param in model.named_parameters():
                                    if L2[0] == 't':
                                        param.grad = param.grad + lamd * param.data
                                    param.data = param.data - gx_lr * param.grad

                            elif optimizer == 'momentum':
                                fm = fm * beta
                                x, y = x.to(device), y.to(device)  # 将输入数据移动到GPU上
                                output = model(x)  # 前向传播
                                loss = criterion(output.view(-1, vocab_size), y.view(-1))  # 求损失
                                # loss = torch.exp(criterion(output.view(-1, vocab_size), y.view(-1)))  # 求损失
                                model.zero_grad()  # 将模型中所有参数的梯度清零
                                loss.backward()  # 反向传播求梯度
                                for name, param in model.named_parameters():
                                    if L2[0] == 't':
                                        param.grad = param.grad + lamd * param.data
                                    m[name] = (1 - beta) * param.grad + beta * m[name]
                                    param.data = param.data - gx_lr * (1 / (1 - fm)) * m[name]

                            elif optimizer == 'NAG':
                                fm = fm * beta
                                x, y = x.to(device), y.to(device)  # 将输入数据移动到GPU上
                                for name, param in model.named_parameters():
                                    param.data = param.data - gx_lr * beta * (1 / (1 - fm)) * m[name]
                                output = model(x)  # 前向传播
                                loss = criterion(output.view(-1, vocab_size), y.view(-1))  # 求损失
                                # loss = torch.exp(criterion(output.view(-1, vocab_size), y.view(-1)))  # 求损失
                                model.zero_grad()  # 将模型中所有参数的梯度清零
                                loss.backward()  # 反向传播求梯度

                                for name, param in model.named_parameters():
                                    if L2[0] == 't':
                                        param.grad = param.grad + lamd * param.data
                                    m[name] = (1 - beta) * param.grad + beta * m[name]
                                    param.data = lin_w[name] - gx_lr * (1 / (1 - fm)) * m[name]
                                    # 记录当前模型参数
                                    lin_w[name] = param.data

                            elif optimizer == 'RNAG':
                                fm = fm * beta
                                x, y = x.to(device), y.to(device)  # 将输入数据移动到GPU上
                                output = model(x)  # 前向传播
                                loss = criterion(output.view(-1, vocab_size), y.view(-1))  # 求损失
                                # loss = torch.exp(criterion(output.view(-1, vocab_size), y.view(-1)))  # 求损失
                                model.zero_grad()  # 将模型中所有参数的梯度清零
                                loss.backward()  # 反向传播求梯度
                                for name, param in model.named_parameters():
                                    if L2[0] == 't':
                                        param.grad = param.grad + lamd * param.data
                                    m[name] = (1 - beta) * param.grad + beta * m[name]
                                    param.data = param.data - gx_lr * (
                                                (1 - beta) * param.grad + (beta / (1 - fm)) * m[name])

                            else:
                                print("geng xin chu cuo!")
                            if (i + 1) % 50 == 0:
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
                                # loss = torch.exp(criterion(output.view(-1, vocab_size), y.view(-1)))  # 求损失
                                total_loss += loss

                            ave_loss = total_loss / len(train_loader)
                            train_loss_list.append(float(ave_loss.item()))

                            print('epoch{} loss{} on the train set'
                                  .format(epoch + 1, float(ave_loss.item())))

                            # 评估测试集
                            total_loss = 0
                            for k, (x, y) in enumerate(validation_loader):
                                # 将输入数据移动到GPU上
                                x, y = x.to(device), y.to(device)
                                output = model(x)
                                loss = criterion(output.view(-1, vocab_size), y.view(-1))  # 求损失
                                # loss = torch.exp(criterion(output.view(-1, vocab_size), y.view(-1)))  # 求损失
                                total_loss += loss

                            ave_loss = total_loss / len(validation_loader)
                            validation_loss_list.append(float(ave_loss.item()))

                            print('epoch{} loss{} on the test set'
                                  .format(epoch + 1, float(ave_loss.item())))

                    # 将列表保存到文件
                    # 拼接文件路径
                    file = '../save_data/{}_{}_{}/{}_beta{}_v{}_lr{}_seed{}.pkl' \
                        .format(file_name, model_name, dataset_name, optimizer, beta, v, lr, seed_index)

                    with open(file, 'wb') as f:
                        pickle.dump({
                            'train_loss_list': train_loss_list,
                            'validation_loss_list': validation_loss_list,
                        }, f)

                    print("Train metrics saved successfully.")

                all_train_loss = np.zeros(Epoch)
                all_validation_loss = np.zeros(Epoch)

                k = len(seed_list)

                for p in range(k):
                    file = '../save_data/{}_{}_{}/{}_beta{}_v{}_lr{}_seed{}.pkl' \
                        .format(file_name, model_name, dataset_name, optimizer, beta, v, lr, p)
                    with open(file, 'rb') as f:
                        data = pickle.load(f)
                    train_loss = np.array(data['train_loss_list'])
                    validation_loss = np.array(data['validation_loss_list'])

                    all_train_loss += train_loss
                    all_validation_loss += validation_loss

                ave_train_loss = all_train_loss / k
                ave_validation_loss = all_validation_loss / k

                ave_train_loss = ave_train_loss.tolist()
                ave_validation_loss = ave_validation_loss.tolist()

                file = '../save_data/{}_{}_{}/ave_{}_beta{}_v{}_lr{}.pkl'\
                    .format(file_name, model_name, dataset_name, optimizer, beta, v, lr)

                with open(file, 'wb') as f:
                    pickle.dump({
                        'train_loss_list': ave_train_loss,
                        'validation_loss_list': ave_validation_loss,
                    }, f)

                print("save successfully")

                # # 绘制折线图并保存
                if optimizer == 'SGD':
                    plt.figure(figsize=(10, 6))
                    plt.plot(ave_train_loss,
                             label='{} lr{} train'.format(optimizer, lr))
                    plt.plot(ave_validation_loss,
                             label='{} lr{} validation'.format(optimizer, lr))
                    # plt.plot(np.full(Epoch, 9), label='9')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.grid()
                    plt.savefig('../save_tu/{}_{}_{}/{}_beta{}_v{}_lr{}_loss.png'
                                .format(file_name, model_name, dataset_name, optimizer, beta, v, lr))
                    plt.clf()  # 清除图像内容
                    plt.close()  # 关闭图像

                else:
                    SGD_file = f'../save_data/{file_name}_{model_name}_{dataset_name}/ave_SGD_beta0.9_v1_lr1.pkl'
                    with open(SGD_file, 'rb') as f:
                        SGD_data = pickle.load(f)
                    SGD_train_loss = SGD_data['train_loss_list']
                    SGD_validation_loss = SGD_data['validation_loss_list']

                    plt.figure(figsize=(10, 6))
                    plt.plot(SGD_train_loss,
                             label='SGD train loss')
                    plt.plot(SGD_validation_loss,
                             label='SGD validation loss')
                    plt.plot(ave_train_loss,
                             label='{} β{} v{} train loss'.format(optimizer, beta, v))
                    plt.plot(ave_validation_loss,
                             label='{} β{} v{} validation loss'.format(optimizer, beta, v))
                    # plt.plot(np.full(Epoch, 6), label='6')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.grid()
                    plt.savefig('../save_tu/{}_{}_{}/{}_beta{}_v{}_lr{}_loss.png'
                                .format(file_name, model_name, dataset_name, optimizer, beta, v, lr))
                    plt.clf()  # 清除图像内容
                    plt.close()  # 关闭图像

                print("tu saved successfully.")
