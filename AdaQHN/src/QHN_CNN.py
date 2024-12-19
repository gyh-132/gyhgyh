# SGD, momentum, NAG, RNAG, QHM, QHN
# QHM可恢复SGD(beta=0, v=1)、moment(beta不等于0, v=1)、R_NAG(beta任意, v=beta)
# QHN可恢复SGD(beta=0, v=1)、NAG(beta不等于0, v = 1)

import torch
import torch.nn as nn
import numpy as np
import Model
import pickle
import matplotlib.pyplot as plt
import os
from data import dataloader

# 导入超参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device= ', device)
# 实验需求超参数
model_name, dataset_name, optimizer = 'GLeNet', 'FMnist', 'SGD'  # 模型、数据集与优化器
model_seed, data_seed = 42, 42  # 模型参数初始化与数据集分批所需随机种子
lr, Epoch, batch_size = 0.1, 80, 128  # 学习率、周期、样本分批大小

# 按需求修改
# seed_list = torch.randint(10, 1000, (3, 2)).numpy()
# print('seed_list:', seed_list)

# seed_list = [[825, 831], [253, 995], [192, 418], [629, 808], [390, 427]]
# bs_list = [64, 128, 256]
# lr_list = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  # SGD
# lr_list = [0.1, 0.2, 0.3, 0.5, 0.7, 1]  # 其它
# beta_list = [0.8, 0.9, 0.95, 0.99, 0.995]  # Momentum、NAG、RNAG
# beta_list = [0.9, 0.95, 0.99, 0.995, 0.999, 0.9995]  # QHM、QHN
# v_list = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
# opt_list = ['SGD', 'momentum', 'NAG', 'RNAG', 'QHM', 'QHN']
lr_list = [0.1, 0.2, 0.3, 0.5, 0.7, 1]
seed_list = [[825, 831], [253, 995], [192, 418]]
opt_list = ['QHM', 'QHN']

L2 = ['t', 1e-4]      # L2正则    分别表示：是否启用、正则项系数
yr = ['t', 10, 0.1]   # 学习率预热 分别表示：是否启用、预热周期、初始学习率与固定学习率的比例
TH = ['t', 30, 0.2]     # 学习率退火 分别表示：是否启用、执行退火的间隔、退火系数

DB1 = np.full(Epoch, 0.05)
# DB2 = np.full(Epoch, 0.3)
DB3 = np.full(Epoch, 1)
# DB4 = np.full(Epoch, 0.93)
# 开始实验
for lr_index in range(len(lr_list)):
    lr = lr_list[lr_index]

    for opt in range(len(opt_list)):
        optimizer = opt_list[opt]

        if optimizer == 'SGD':
            beta_list = [0.9]
            v_list = [1]
        elif optimizer in ['momentum', 'NAG', 'RNAG']:
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

                if optimizer != 'SGD':
                    SGD_file = f'../save_data/QHN_GLeNet_FMnist/ave_SGD_beta0.9_v1_lr1.pkl'
                    with open(SGD_file, 'rb') as f:
                        SGD_data = pickle.load(f)
                    SGD_train_loss = SGD_data['train_loss_list']
                    SGD_validation_loss = SGD_data['validation_loss_list']
                    SGD_train_acc = SGD_data['train_acc_list']
                    SGD_validation_acc = SGD_data['validation_acc_list']

                for seed_index in range(len(seed_list)):
                    model_seed = seed_list[seed_index][0]
                    data_seed = seed_list[seed_index][1]

                    print('model:{}, dataset:{}, model_seed:{}, data_seed:{}' \
                          .format(model_name, dataset_name, model_seed, data_seed))
                    print('lr:{}, Epoch:{}, batch_size:{}'.format(lr, Epoch, batch_size))
                    print('optimizer:{}, beta:{}, v:{}'.format(optimizer, beta, v))
                    print('L2:{}, yr:{}, TH:{}'.format(L2, yr, TH))

                    # 导入数据集
                    train_loader, validation_loader = dataloader(data_seed, batch_size, dataset_name)
                    # 定义模型、损失函数
                    if model_name == 'LeNet':
                        model = Model.LeNet(model_seed).to(device)
                    elif model_name == 'GLeNet':
                        model = Model.GLeNet(model_seed).to(device)
                    else:
                        print("model cuo wu!")
                    criterion = nn.CrossEntropyLoss()

                    # 储存实验数据
                    # 记录每一epoch后模型在训练集和验证集上的损失与准确率
                    train_loss_list = []
                    train_acc_list = []
                    validation_loss_list = []
                    validation_acc_list = []

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
                                loss = criterion(output, y)  # 求损失
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
                                loss = criterion(output, y)  # 求损失
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
                                loss = criterion(output, y)  # 求损失
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
                                loss = criterion(output, y)  # 求损失
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
                                loss = criterion(output, y)  # 求损失
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
                                loss = torch.sqrt(criterion(output, y))  # 求均方根损失
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
                            if (i + 1) % 100 == 0:
                                print(
                                    f'Epoch [{epoch + 1}], iteration [{epoch * len(train_loader) + i + 1}], Loss: {loss.item():.4f}')
                        # 模型设置为评估模式
                        model.eval()

                        # 评估训练集
                        total_loss = 0
                        total_correct = 0
                        total_samples = 0

                        with torch.no_grad():
                            for j, (x, y) in enumerate(train_loader):
                                # 将输入数据移动到GPU上
                                x, y = x.to(device), y.to(device)
                                output = model(x)

                                loss = criterion(output, y)
                                total_loss += loss

                                _, predicted = torch.max(output, 1)
                                total_correct += (predicted == y).sum().item()
                                total_samples += y.size(0)

                            accuracy = total_correct / total_samples
                            ave_loss = total_loss / len(train_loader)

                            train_loss_list.append(float(ave_loss.item()))
                            train_acc_list.append(float(accuracy))

                            print('epoch{} loss{} Accuracy on the train set: {:.2f}%'.format(epoch + 1,
                                                                                             float(ave_loss.item()),
                                                                                             accuracy * 100))

                        # 评估测试集
                        total_loss = 0
                        total_correct = 0
                        total_samples = 0

                        with torch.no_grad():
                            for k, (x, y) in enumerate(validation_loader):
                                # 将输入数据移动到GPU上
                                x, y = x.to(device), y.to(device)
                                output = model(x)

                                loss = criterion(output, y)
                                total_loss += loss

                                _, predicted = torch.max(output, 1)
                                total_correct += (predicted == y).sum().item()
                                total_samples += y.size(0)

                            accuracy = total_correct / total_samples
                            ave_loss = total_loss / len(validation_loader)

                            validation_loss_list.append(float(ave_loss.item()))
                            validation_acc_list.append(float(accuracy))

                            print(
                                'epoch{} loss{} Accuracy on the test set: {:.2f}%'.format(epoch + 1,
                                                                                          float(ave_loss.item()),
                                                                                          accuracy * 100))

                    # 将列表保存到文件
                    # 拼接文件路径
                    file_name = '../save_data/QHN_{}_{}/{}_beta{}_v{}_lr{}_seed{}.pkl' \
                        .format(model_name, dataset_name, optimizer, beta, v, lr, seed_index)

                    with open(file_name, 'wb') as f:
                        pickle.dump({
                            'train_loss_list': train_loss_list,
                            'train_acc_list': train_acc_list,
                            'validation_loss_list': validation_loss_list,
                            'validation_acc_list': validation_acc_list,
                        }, f)

                    print("Train metrics saved successfully.")

                all_train_loss = np.zeros(Epoch)
                all_train_acc = np.zeros(Epoch)
                all_validation_loss = np.zeros(Epoch)
                all_validation_acc = np.zeros(Epoch)

                k = len(seed_list)

                for p in range(k):
                    file = '../save_data/QHN_{}_{}/{}_beta{}_v{}_lr{}_seed{}.pkl' \
                        .format(model_name, dataset_name, optimizer, beta, v, lr, p)
                    with open(file, 'rb') as f:
                        data = pickle.load(f)
                    train_loss = np.array(data['train_loss_list'])
                    train_acc = np.array(data['train_acc_list'])
                    validation_loss = np.array(data['validation_loss_list'])
                    validation_acc = np.array(data['validation_acc_list'])

                    all_train_loss += train_loss
                    all_train_acc += train_acc
                    all_validation_loss += validation_loss
                    all_validation_acc += validation_acc

                ave_train_loss = all_train_loss / k
                ave_train_acc = all_train_acc / k
                ave_validation_loss = all_validation_loss / k
                ave_validation_acc = all_validation_acc / k

                ave_train_loss = ave_train_loss.tolist()
                ave_train_acc = ave_train_acc.tolist()
                ave_validation_loss = ave_validation_loss.tolist()
                ave_validation_acc = ave_validation_acc.tolist()

                file_name = '../save_data/QHN_{}_{}/ave_{}_beta{}_v{}_lr{}.pkl'.format(model_name, dataset_name,
                                                                                       optimizer, beta, v, lr)

                with open(file_name, 'wb') as f:
                    pickle.dump({
                        'train_loss_list': ave_train_loss,
                        'train_acc_list': ave_train_acc,
                        'validation_loss_list': ave_validation_loss,
                        'validation_acc_list': ave_validation_acc,
                    }, f)

                print("save successfully")

                # # 绘制折线图并保存
                if optimizer == 'SGD':
                    plt.figure(figsize=(10, 6))
                    plt.plot(ave_train_loss,
                             label='{} β{} v{} train'.format(optimizer, beta, v))
                    plt.plot(ave_validation_loss,
                             label='{} β{} v{} validation'.format(optimizer, beta, v))
                    plt.plot(DB1, label='0.05')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.grid()
                    plt.savefig(
                        '../save_tu/QHN_{}_{}/{}_beta{}_v{}_lr{}_loss.png'.format(model_name, dataset_name, optimizer,
                                                                                  beta, v, lr))
                    plt.clf()  # 清除图像内容
                    plt.close()  # 关闭图像

                    # # 绘制折线图并保存
                    plt.figure(figsize=(10, 6))
                    plt.plot(ave_train_acc,
                             label='{} β{} v{} train '.format(optimizer, beta, v))
                    plt.plot(ave_validation_acc,
                             label='{} β{} v{} validation '.format(optimizer, beta, v))
                    plt.plot(DB3, label='1')
                    plt.xlabel('Epoch')
                    plt.ylabel('Acc')
                    plt.legend()
                    plt.grid()
                    plt.savefig(
                        '../save_tu/QHN_{}_{}/{}_beta{}_v{}_lr{}_acc.png'.format(model_name, dataset_name, optimizer,
                                                                                 beta, v, lr))
                    plt.clf()  # 清除图像内容
                    plt.close()  # 关闭图像
                else:
                    plt.figure(figsize=(10, 6))
                    plt.plot(SGD_train_loss,
                             label='SGD train loss')
                    plt.plot(SGD_validation_loss,
                             label='SGD validation loss')
                    plt.plot(ave_train_loss,
                             label='{} β{} v{} train loss'.format(optimizer, beta, v))
                    plt.plot(ave_validation_loss,
                             label='{} β{} v{} validation loss'.format(optimizer, beta, v))
                    plt.plot(DB1, label='0.05')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.grid()
                    plt.savefig(
                        '../save_tu/QHN_{}_{}/{}_beta{}_v{}_lr{}_loss.png'.format(model_name, dataset_name, optimizer,
                                                                                  beta, v, lr))
                    plt.clf()  # 清除图像内容
                    plt.close()  # 关闭图像

                    # # 绘制折线图并保存
                    plt.figure(figsize=(10, 6))
                    plt.plot(SGD_train_acc,
                             label='SGD train acc')
                    plt.plot(SGD_validation_acc,
                             label='SGD validation acc')
                    plt.plot(ave_train_acc,
                             label='{} β{} v{} train '.format(optimizer, beta, v))
                    plt.plot(ave_validation_acc,
                             label='{} β{} v{} validation '.format(optimizer, beta, v))
                    plt.plot(DB3, label='1')
                    plt.xlabel('Epoch')
                    plt.ylabel('Acc')
                    plt.legend()
                    plt.grid()
                    plt.savefig(
                        '../save_tu/QHN_{}_{}/{}_beta{}_v{}_lr{}_acc.png'.format(model_name, dataset_name, optimizer,
                                                                                 beta, v, lr))
                    plt.clf()  # 清除图像内容
                    plt.close()  # 关闭图像

                print("tu saved successfully.")

