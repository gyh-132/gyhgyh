import torch
import torch.nn as nn
import numpy as np
import Model
import pickle
import matplotlib.pyplot as plt
import matplotlib
from data import dataloader
import itertools
import os
import time

# 实验需求超参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device= ', device)

file_name = 'AdaQHN'
# model_name, dataset_name = ['PoolMLP', 'EMnist_digits']; ['GLeNet5', 'FMnist']; ['TinyVGG', 'EMnist_letters'];
# ['MicroVGG', 'cifar10']; ['LSTM1', 'AGnews']; ['GRU1', 'AGnews']
model_name, dataset_name = 'PoolMLP', 'EMnist_digits'  # 模型、数据集
Epoch, batch_size = 80, 128  # 周期、样本分批大小
# opt_list = ['Adam', 'Adam_win', 'Adan', 'QHAdam', 'AdaQHN']
opt_list = ['Adan', 'QHAdam', 'AdaQHN']  # 优化算法
# # seed_list = torch.randint(10, 1000, (5, 2)).numpy()
# # print('seed_list:', seed_list)
# seed_list = [[825, 831], [253, 995], [192, 418], [629, 808], [390, 427]]
seed_list = [[825, 831], [253, 995], [192, 418]]  # 模型参数初始化与数据集分批所需随机种子

"""
由于学习率lr对算法性能影响较大，因此首先在各算法默认参数（参考原论文）下对lr进行一次粗略搜索以确定最佳lr可能落在的区间，
然后对所有算法进行第二次参数搜索（使用网格搜索），参数包括lr和算法本身所需的超参数。

第一次lr搜索范围（粗略的）= [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]

各算法默认超参数分别为 Adam[0.9, 0.999]; Adam_win[0.9, 0.999]; Adan[0.96, 0.93, 0.96]
QHAdam[0.995, 0.999, 0.7]; AdaQHN[0.995, 0.999, 0.7]

各算法超参数搜索范围分别为Adam[[0.8, 0.85, 0.9, 0.95, 0.99], 0.999]; Adam_win[[0.8, 0.85, 0.9, 0.95, 0.99], 0.999];
Adan[[0.9, 0.93, 0.96, 0.99], [0.9, 0.93, 0.96, 0.99], [0.9, 0.93, 0.96, 0.99]]
QHAdam[[0.95, 0.99, 0.995, 0.999, 0.9995], 0.999, [0.5, 0.6, 0.7, 0.8, 0.9]];
AdaQHN[[0.95, 0.99, 0.995, 0.999, 0.9995], 0.999, [0.5, 0.6, 0.7, 0.8, 0.9]]

在['PoolMLP', 'EMnist_digits']，第二次lr搜索范围为  Adam、Adam_win[0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05];
和Adan、QHAdam、AdaQHN[0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07]

在['GLeNet5', 'FMnist']上，第二次lr搜索范围均为[0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03]；

在['TinyVGG', 'EMnist_letters']上，第二次lr搜索范围均为[0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05]

在['MicroVGG', 'cifar10']上，第二次lr搜索范围分别为 Adam, Adam_win, Adan[0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05];
QHAdam和AdaQHN[0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07]

在['GRU1', 'AGnews']上，第二次lr搜索范围均为[0.0007, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.007]

在['LSTM1', 'AGnews']上，第二次lr搜索范围分别为 
Adam、Adam_win[0.0001, 0.00015, 0.0002, 0.0003, 0.0005, 0.0007, 0.001]([0.0002, 0.0003, 0.0005, 0.0007, 0.001]);
Adan[0.0005, 0.0007, 0.001, 0.0015, 0.002, 0.003, 0.005]([0.0007, 0.001, 0.0015, 0.002, 0.003]);
QHMAdam和AdaQHN[0.001, 0.0015, 0.002, 0.003, 0.005, 0.007, 0.01]
"""

L2 = ['t', 1e-4]      # L2正则    分别表示：是否启用、正则项系数
yr = ['t', 10, 0.1]   # 学习率预热 分别表示：是否启用、预热周期、初始学习率与固定学习率的比例
TH = ['t', 30, 0.2]     # 学习率退火 分别表示：是否启用、执行退火的间隔、退火系数

# 开始实验
for opt_index in range(len(opt_list)):
    optimizer = opt_list[opt_index]

    if optimizer == 'Adam':
        beta1_list = [0.8, 0.85, 0.9, 0.95, 0.99]
        beta2_list = [0.999]
        parameters_list = [beta1_list, beta2_list]
        lr_list = [0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05]
    elif optimizer == 'Adam_win':
        beta1_list = [0.8, 0.85, 0.9, 0.95, 0.99]
        beta2_list = [0.999]
        parameters_list = [beta1_list, beta2_list]
        lr_list = [0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05]
    elif optimizer == 'Adan':
        beta1_list = [0.9, 0.93, 0.96, 0.99]
        beta2_list = [0.9, 0.93, 0.96, 0.99]
        beta3_list = [0.9, 0.93, 0.96, 0.99]
        parameters_list = [beta1_list, beta2_list, beta3_list]
        lr_list = [0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1]
    elif optimizer == 'QHAdam':
        beta1_list = [0.95, 0.99, 0.995, 0.999, 0.9995]
        beta2_list = [0.999]
        v_list = [0.5, 0.6, 0.7, 0.8, 0.9]
        parameters_list = [beta1_list, beta2_list, v_list]
        lr_list = [0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07]
    elif optimizer == 'AdaQHN':
        beta1_list = [0.95, 0.99, 0.995, 0.999, 0.9995]
        beta2_list = [0.999]
        v_list = [0.5, 0.6, 0.7, 0.8, 0.9]
        parameters_list = [beta1_list, beta2_list, v_list]
        lr_list = [0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07]
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    for parameters in itertools.product(*parameters_list):

        if optimizer in ['Adam', 'Adam_win']:
            beta1, beta2 = parameters
            wd = 1e-4  # wd为权重衰减系数 weight decay，注意Adam没有用到这个参数
        elif optimizer == 'Adan':
            beta1, beta2, beta3 = parameters
            wd = 1e-4  # wd为权重衰减系数 weight decay
        elif optimizer in ['QHAdam', 'AdaQHN']:
            beta1, beta2, v = parameters
        else:
            raise ValueError(f"Invalid parameter set for optimizer: {optimizer}")

        for lr_index in range(len(lr_list)):
            lr = lr_list[lr_index]

            for seed_index in range(len(seed_list)):
                model_seed = seed_list[seed_index][0]
                data_seed = seed_list[seed_index][1]

                print('model:{}, dataset:{}'.format(model_name, dataset_name))
                print('optimizer:{}, parameters:{}, lr:{}'.format(optimizer, parameters, lr))
                print('model_seed:{}, data_seed:{}'.format(model_seed, data_seed))

                # 导入数据集
                # start_time = time.time()
                train_loader, validation_loader = dataloader(data_seed, batch_size, dataset_name)
                # end_time = time.time()
                # print(f"代码运行时间: {end_time - start_time} 秒")
                # 定义模型、损失函数
                if model_name == 'PoolMLP':  # 用于 EMnist_digits
                    model = Model.PoolMLP(model_seed).to(device)
                    criterion = nn.CrossEntropyLoss()
                elif model_name == 'LSTM1':  # 用于 AG News
                    model = Model.LSTM1(model_seed, 12566, 128, 128, 4).to(device)
                    criterion = nn.CrossEntropyLoss()
                elif model_name == 'GRU1':  # 用于 AG News
                    model = Model.GRU1(model_seed, 12566, 128, 128, 4).to(device)
                    criterion = nn.CrossEntropyLoss()
                elif model_name == 'GLeNet5':  # 用于 FMnist
                    model = Model.GLeNet5(model_seed).to(device)
                    criterion = nn.CrossEntropyLoss()
                elif model_name == 'TinyVGG':  # 用于 EMnist_letters
                    model = Model.TinyVGG(model_seed).to(device)
                    criterion = nn.CrossEntropyLoss()
                elif model_name == 'MicroVGG':  # 用于 cifar10
                    model = Model.MicroVGG(model_seed).to(device)
                    criterion = nn.CrossEntropyLoss()
                elif model_name == 'VGG_cifar10':
                    model = Model.VGG_cifar10(model_seed).to(device)
                    criterion = nn.CrossEntropyLoss()
                elif model_name == 'ResNet_cifar10':
                    model = Model.ResNet_cifar10(model_seed).to(device)
                    criterion = nn.CrossEntropyLoss()
                else:
                    raise ValueError(f"Unknown Model: {model_name}")

                # 记录训练时模型在每一epoch上的平均损失和平均准确率
                train_loss_list = []
                train_acc_list = []
                # 记录模型在每一epoch结束后在验证集上的损失与准确率
                validation_loss_list = []
                validation_acc_list = []

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
                lin_w = {}  # 记录额外的模型参数(win中第一个参数序列)
                m2 = {}  # 记录Adan中的vk
                g = {}  # 记录Adan中过去的梯度
                for name, param in model.named_parameters():
                    m[name] = torch.zeros_like(param)
                    h[name] = torch.zeros_like(param)
                    lin_m[name] = torch.zeros_like(param)
                    lin_h[name] = torch.zeros_like(param)
                    lin_w[name] = param.data
                    m2[name] = torch.zeros_like(param)
                    g[name] = torch.zeros_like(param)

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
                    total_loss = 0
                    total_correct = 0
                    total_samples = 0
                    for i, (x, y) in enumerate(train_loader):
                        # 更新模型参数
                        if optimizer == 'Adam':
                            fm = fm * beta1
                            fh = fh * beta2
                            x, y = x.to(device), y.to(device)  # 将输入数据移动到GPU上
                            output = model(x)  # 前向传播
                            loss = criterion(output, y)  # 求损失
                            model.zero_grad()  # 将模型中所有参数的梯度清零
                            loss.backward()  # 反向传播求梯度
                            for name, param in model.named_parameters():
                                if L2[0] == 't':
                                    param.grad = param.grad + lamd * param.data
                                # 更新动量项和自适应步长项
                                m[name] = (1 - beta1) * param.grad + beta1 * m[name]
                                h[name] = (1 - beta2) * (param.grad ** 2) + beta2 * h[name]
                                # 修正动量项和自适应步长项
                                lin_m[name] = m[name] / (1 - fm)
                                lin_h[name] = eps + ((h[name] / (1 - fh)) ** 0.5)
                                # 更新模型参数
                                param.data = param.data - gx_lr * lin_m[name] / lin_h[name]

                        elif optimizer == 'QHAdam':
                            fm = fm * beta1
                            fh = fh * beta2
                            x, y = x.to(device), y.to(device)  # 将输入数据移动到GPU上
                            output = model(x)  # 前向传播
                            loss = criterion(output, y)  # 求损失
                            model.zero_grad()  # 将模型中所有参数的梯度清零
                            loss.backward()  # 反向传播求梯度
                            for name, param in model.named_parameters():
                                if L2[0] == 't':
                                    param.grad = param.grad + lamd * param.data
                                # 更新动量项和自适应步长项
                                m[name] = (1 - beta1) * param.grad + beta1 * m[name]
                                h[name] = (1 - beta2) * (param.grad ** 2) + beta2 * h[name]
                                # 修正动量项和自适应步长项
                                lin_m[name] = (v / (1 - fm)) * m[name] + (1 - v) * param.grad
                                lin_h[name] = eps + ((h[name] / (1 - fh)) ** 0.5)
                                # 更新模型参数
                                param.data = param.data - gx_lr * lin_m[name] / lin_h[name]

                        elif optimizer == 'Adam_win':
                            x, y = x.to(device), y.to(device)  # 将输入数据移动到GPU上
                            output = model(x)  # 前向传播
                            loss = criterion(output, y)  # 求损失
                            model.zero_grad()  # 将模型中所有参数的梯度清零
                            loss.backward()  # 反向传播求梯度
                            # 按论文要求对动量项和自适应步长项赋初值
                            if epoch == 0 and i == 0:
                                for name, param in model.named_parameters():
                                    m[name] = param.grad
                                    h[name] = param.grad ** 2
                            for name, param in model.named_parameters():
                                if L2[0] == 't':
                                    param.grad = param.grad + lamd * param.data
                                # 更新动量项和自适应步长项
                                m[name] = (1 - beta1) * param.grad + beta1 * m[name]
                                h[name] = (1 - beta2) * (param.grad ** 2) + beta2 * h[name]
                                # 动量项和自适应步长项的结合 u_k
                                lin_m[name] = m[name] / (eps + (h[name] ** 0.5))
                                # 更新第一个参数序列和模型参数序列（按论文 默认第二个步长等于第一个步长的两倍，即n’=2n）
                                lin_w[name] = (lin_w[name] - gx_lr * lin_m[name]) / (1 + wd * gx_lr)
                                param.data = (2 * lin_w[name] + (param.data - 2 * gx_lr * lin_m[name])) / (
                                        3 + 2 * wd * gx_lr)

                        elif optimizer == 'Adan':
                            x, y = x.to(device), y.to(device)  # 将输入数据移动到GPU上
                            output = model(x)  # 前向传播
                            loss = criterion(output, y)  # 求损失
                            model.zero_grad()  # 将模型中所有参数的梯度清零
                            loss.backward()  # 反向传播求梯度
                            # 按论文要求对动量项和自适应步长项赋初值
                            if epoch == 0 and i == 0:
                                for name, param in model.named_parameters():
                                    m[name] = param.grad
                                    h[name] = param.grad ** 2
                            for name, param in model.named_parameters():
                                if L2[0] == 't':
                                    param.grad = param.grad + lamd * param.data
                                # 更新动量项和自适应步长项并记录梯度
                                m[name] = (1 - beta1) * param.grad + beta1 * m[name]
                                m2[name] = (1 - beta2) * (param.grad - g[name]) + beta2 * m2[name]
                                h[name] = (1 - beta3) * ((param.grad + beta2 * (param.grad - g[name])) ** 2) + beta3 * \
                                          h[name]
                                g[name] = param.grad
                                # 修正动量项和自适应步长项
                                lin_m[name] = m[name] + beta2 * m2[name]
                                lin_h[name] = eps + (h[name] ** 0.5)
                                # 更新模型参数
                                param.data = (param.data - gx_lr * lin_m[name] / lin_h[name]) / (1 + gx_lr * wd)

                        elif optimizer == 'AdaQHN':
                            fm = fm * beta1
                            fh = fh * beta2
                            x, y = x.to(device), y.to(device)  # 将输入数据移动到GPU上
                            # 将模型参数外推用以计算外推梯度
                            for name, param in model.named_parameters():
                                param.data = param.data - gx_lr * (v / (1 - fm)) * beta1 * m[name]
                            output = model(x)  # 前向传播
                            loss = criterion(output, y)  # 求损失
                            model.zero_grad()  # 将模型中所有参数的梯度清零
                            loss.backward()  # 反向传播求梯度
                            for name, param in model.named_parameters():
                                if L2[0] == 't':
                                    param.grad = param.grad + lamd * param.data
                                param.data = param.data - gx_lr * (v / (1 - fm)) * beta1 * m[name]
                                # 更新自适应步长项并修正当前外推梯度
                                h[name] = (1 - beta2) * (param.grad ** 2) + beta2 * h[name]
                                param.grad = param.grad / (eps + ((h[name] / (1 - fh)) ** 0.5))
                                # 更新并修正动量项
                                m[name] = (1 - beta1) * param.grad + beta1 * m[name]
                                lin_m[name] = (v / (1 - fm)) * m[name] + (1 - v) * param.grad
                                # 更新模型参数并记录
                                param.data = param.data - gx_lr * lin_m[name]
                        else:
                            raise ValueError(f"Unknown optimizer: {optimizer}")

                        total_loss += loss
                        _, predicted = torch.max(output, 1)
                        total_correct += (predicted == y).sum().item()
                        total_samples += y.size(0)

                    accuracy = total_correct / total_samples
                    ave_loss = total_loss / len(train_loader)
                    train_loss_list.append(float(ave_loss.item()))
                    train_acc_list.append(float(accuracy))
                    print(f'epoch{epoch+1} train set: loss={float(ave_loss.item()):.6f}, Accuracy={accuracy * 100:.2f}')

                    # 模型设置为评估模式
                    model.eval()
                    with torch.no_grad():
                        # 评估验证集
                        total_loss = 0
                        total_correct = 0
                        total_samples = 0

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

                        print(f'epoch{epoch + 1} validation set: loss={float(ave_loss.item()):.6f}, '
                              f'Accuracy={accuracy * 100:.2f}')
                # 构建路径
                save_data_dir = f'../save_data/{file_name}_{model_name}_{dataset_name}/'
                save_tu_loss_dir = f'../save_tu/{file_name}_{model_name}_{dataset_name}/loss/'
                save_tu_acc_dir = f'../save_tu/{file_name}_{model_name}_{dataset_name}/acc/'
                # 检查路径是否存在，不存在则创建
                os.makedirs(save_data_dir, exist_ok=True)
                os.makedirs(save_tu_loss_dir, exist_ok=True)
                os.makedirs(save_tu_acc_dir, exist_ok=True)

                # 保存实验数据
                file = f'../save_data/{file_name}_{model_name}_{dataset_name}/{optimizer}_{parameters}_{lr}_{seed_index}.pkl'
                with open(file, 'wb') as f:
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

            L = len(seed_list)

            for p in range(L):
                file = f'../save_data/{file_name}_{model_name}_{dataset_name}/{optimizer}_{parameters}_{lr}_{p}.pkl'
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

            ave_train_loss = all_train_loss / L
            ave_train_acc = all_train_acc / L
            ave_validation_loss = all_validation_loss / L
            ave_validation_acc = all_validation_acc / L

            ave_train_loss = ave_train_loss.tolist()
            ave_train_acc = ave_train_acc.tolist()
            ave_validation_loss = ave_validation_loss.tolist()
            ave_validation_acc = ave_validation_acc.tolist()

            file = f'../save_data/{file_name}_{model_name}_{dataset_name}/ave_{optimizer}_{parameters}_{lr}.pkl'
            with open(file, 'wb') as f:
                pickle.dump({
                    'train_loss_list': ave_train_loss,
                    'train_acc_list': ave_train_acc,
                    'validation_loss_list': ave_validation_loss,
                    'validation_acc_list': ave_validation_acc,
                }, f)
            print("ave data save successfully")

        colors = matplotlib.colormaps.get_cmap('tab10')  # 颜色映射
        plt.figure(figsize=(10, 6))
        for index in range(len(lr_list)):
            file = f'../save_data/{file_name}_{model_name}_{dataset_name}/ave_{optimizer}_{parameters}_{lr_list[index]}.pkl'
            with open(file, 'rb') as f:
                data = pickle.load(f)
            train_loss = np.array(data['train_loss_list'])
            # validation_loss = np.array(data['validation_loss_list'])

            x_values = list(range(20, len(train_loss)))
            plt.plot(x_values, train_loss[20:], label=f'{optimizer} parameters{parameters} lr{lr_list[index]}', color=colors(index))
            # plt.plot(x_values, validation_loss[5:], label=f'{optimizer} parameters{parameters} lr{lr_list[index]}', linestyle='--', color=colors(index))

        # plt.plot(np.full(Epoch, 0), label='0')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.savefig(f'../save_tu/{file_name}_{model_name}_{dataset_name}/loss/{optimizer}_{parameters}_loss.png')
        plt.clf()  # 清除图像内容
        plt.close()  # 关闭图像

        plt.figure(figsize=(10, 6))
        for index in range(len(lr_list)):
            file = f'../save_data/{file_name}_{model_name}_{dataset_name}/ave_{optimizer}_{parameters}_{lr_list[index]}.pkl'
            with open(file, 'rb') as f:
                data = pickle.load(f)
            # train_acc = np.array(data['train_acc_list'])
            validation_acc = np.array(data['validation_acc_list'])

            x_values = list(range(20, len(train_loss)))
            # plt.plot(x_values, train_acc[40:], label=f'{optimizer} parameters{parameters} lr{lr_list[index]}', color=colors(index))
            plt.plot(x_values, validation_acc[20:], label=f'{optimizer} parameters{parameters} lr{lr_list[index]}', linestyle='--', color=colors(index))
        # plt.plot(np.full(Epoch, 1), label='1')
        plt.xlabel('Epoch')
        plt.ylabel('ACC')
        plt.legend()
        plt.grid()
        plt.savefig(f'../save_tu/{file_name}_{model_name}_{dataset_name}/acc/{optimizer}_{parameters}_acc.png')
        plt.clf()  # 清除图像内容
        plt.close()  # 关闭图像

        print("tu saved successfully.")


