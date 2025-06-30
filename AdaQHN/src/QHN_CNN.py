# SGD, momentum, NAG, QHM, QHN
# QHM可恢复SGD(beta=0, v=1)、moment(beta不等于0, v=1)、R_NAG(beta任意, v=beta)
# QHN可恢复SGD(beta=0, v=1)、NAG(beta不等于0, v = 1)

import torch
import torch.nn as nn
import numpy as np
import os
import Model
import pickle
import matplotlib.pyplot as plt
import matplotlib
from data import dataloader
import itertools

# 实验需求超参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device= ', device)

file_name = 'QHN'
# model_name, dataset_name, = ['PoolMLP', 'EMnist_digits']; ['TinyVGG', 'EMnist_letters'];
# ['MobileCNN', 'EMnist_balanced']; ['GRU1', 'AGnews']
model_name, dataset_name, = ['GRU1', 'AGnews']  # 模型、数据集
Epoch, batch_size = 80, 128  # 周期、样本分批大小
# seed_list = [[825, 831], [253, 995], [192, 418], [629, 808], [390, 427]]
seed_list = [[825, 831], [253, 995], [192, 418]]
# opt_list = ['SGD', 'momentum', 'NAG', 'QHM', 'QHN']
opt_list = ['SGD', 'momentum', 'NAG', 'QHM', 'QHN']

"""
由于学习率lr对算法性能影响较大，因此首先在各算法默认参数（参考原论文）下对lr进行一次粗略搜索以确定最佳lr可能落在的区间，
然后对所有算法进行第二次参数搜索（使用网格搜索），参数包括lr和算法本身所需的超参数。

第一次lr搜索范围（粗略的）= [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
各算法默认参数分别为 momentum[0.9]; NAG[0.9]; QHM[0.9, 0.7]; QHN[0.9, 0.7]
各算法搜索范围分别为momentum,NAG[0.8, 0.85, 0.9, 0.95, 0.99]; 
QHM,QHN[[0.8, 0.85, 0.9, 0.95, 0.99], [0.5, 0.6, 0.7, 0.8, 0.9]]

在['PoolMLP', 'EMnist_digits']上，第二次lr搜索范围分别为SGD[0.5, 0.7, 1, 1.2, 1.5]，其它[2, 2.5, 3, 3.5, 4]

在['TinyVGG', 'EMnist_letters']上，第二次lr搜索范围分别为SGD[0.15, 0.2, 0.3, 0.5, 0.7]，其它[0.7, 1, 1.2, 1.5, 2]

在['MobileCNN', 'EMnist_balanced']上，第二次lr搜索范围分别为SGD[0.2, 0.3, 0.5, 0.7, 1];其它[0.7, 1, 1.2, 1.5, 2]

在['GRU1', 'AGnews']上，第二次lr搜索范围分别为SGD[0.2, 0.3, 0.5, 0.7, 1];其它[0.7, 1, 1.2, 1.5, 2]
"""

L2 = ['t', 1e-4]      # L2正则    分别表·示：是否启用、正则项系数
yr = ['t', 10, 0.1]   # 学习率预热 分别表示：是否启用、预热周期、初始学习率与固定学习率的比例
TH = ['t', 30, 0.2]     # 学习率退火 分别表示：是否启用、执行退火的间隔、退火系数

# 开始实验
for opt_index in range(len(opt_list)):
    optimizer = opt_list[opt_index]

    if optimizer == 'SGD':
        beta_list = [0]
        parameters_list = [beta_list]
        lr_list = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    elif optimizer == 'momentum':
        beta_list = [0.9]
        parameters_list = [beta_list]
        lr_list = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    elif optimizer == 'NAG':
        beta_list = [0.9]
        parameters_list = [beta_list]
        lr_list = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    elif optimizer == 'QHM':
        beta_list = [0.9]
        v_list = [0.7]
        parameters_list = [beta_list, v_list]
        lr_list = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    elif optimizer == 'QHN':
        beta_list = [0.9]
        v_list = [0.7]
        parameters_list = [beta_list, v_list]
        lr_list = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    for parameters in itertools.product(*parameters_list):

        if optimizer in ['SGD', 'momentum', 'NAG']:
            """这里没有直接将parameters赋给beta，是因为当parameters_list只有一个列表时，
            parameters是单元素的元组，此时beta也是元组"""
            beta = parameters[0]
        elif optimizer in ['QHM', 'QHN']:
            beta, v = parameters
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
                train_loader, validation_loader = dataloader(data_seed, batch_size, dataset_name)
                # 定义模型、损失函数
                if model_name == 'PoolMLP':
                    model = Model.PoolMLP(model_seed).to(device)
                    criterion = nn.CrossEntropyLoss()
                elif model_name == 'TinyVGG':
                    model = Model.TinyVGG(model_seed).to(device)
                    criterion = nn.CrossEntropyLoss()
                elif model_name == 'MobileCNN':
                    model = Model.MobileCNN(model_seed).to(device)
                    criterion = nn.CrossEntropyLoss()
                elif model_name == 'GRU1':
                    model = Model.GRU1(model_seed, 12566, 128, 128, 4).to(device)
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
                lamd = L2[1]  # 正则系数
                m = {}  # 记录动量项
                for name, param in model.named_parameters():
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
                    total_loss = 0
                    total_correct = 0
                    total_samples = 0

                    for i, (x, y) in enumerate(train_loader):
                        # 更新模型参数
                        if optimizer == 'QHM':
                            x, y = x.to(device), y.to(device)  # 将输入数据移动到GPU上
                            output = model(x)  # 前向传播
                            loss = criterion(output, y)  # 求损失
                            model.zero_grad()  # 将模型中所有参数的梯度清零
                            loss.backward()  # 反向传播求梯度
                            for name, param in model.named_parameters():
                                if L2[0] == 't':
                                    param.grad = param.grad + lamd * param.data
                                m[name] = (1 - beta) * param.grad + beta * m[name]
                                param.data = param.data - gx_lr * ((1 - v) * param.grad + (v * m[name]))

                        elif optimizer == 'QHN':
                            x, y = x.to(device), y.to(device)  # 将输入数据移动到GPU上
                            # 前推模型参数
                            for name, param in model.named_parameters():
                                param.data = param.data - (gx_lr * v * beta * m[name])
                            output = model(x)  # 前向传播
                            loss = criterion(output, y)  # 求损失
                            model.zero_grad()  # 将模型中所有参数的梯度清零
                            loss.backward()  # 反向传播求梯度
                            for name, param in model.named_parameters():
                                # L2正则化
                                if L2[0] == 't':
                                    param.grad = param.grad + lamd * param.data
                                # 恢复模型参数
                                param.data = param.data + (gx_lr * v * beta * m[name])
                                # 更新动量项
                                m[name] = (1 - beta) * param.grad + beta * m[name]
                                # 更新模型参数
                                param.data = param.data - gx_lr * ((1 - v) * param.grad + (v * m[name]))

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
                            x, y = x.to(device), y.to(device)  # 将输入数据移动到GPU上
                            output = model(x)  # 前向传播
                            loss = criterion(output, y)  # 求损失
                            model.zero_grad()  # 将模型中所有参数的梯度清零
                            loss.backward()  # 反向传播求梯度
                            for name, param in model.named_parameters():
                                if L2[0] == 't':
                                    param.grad = param.grad + lamd * param.data
                                m[name] = (1 - beta) * param.grad + beta * m[name]
                                param.data = param.data - gx_lr * m[name]

                        elif optimizer == 'NAG':
                            x, y = x.to(device), y.to(device)  # 将输入数据移动到GPU上
                            for name, param in model.named_parameters():
                                param.data = param.data - gx_lr * beta * m[name]
                            output = model(x)  # 前向传播
                            loss = criterion(output, y)  # 求损失
                            model.zero_grad()  # 将模型中所有参数的梯度清零
                            loss.backward()  # 反向传播求梯度
                            for name, param in model.named_parameters():
                                # 恢复模型参数
                                param.data = param.data + gx_lr * beta * m[name]
                                if L2[0] == 't':
                                    param.grad = param.grad + lamd * param.data
                                m[name] = (1 - beta) * param.grad + beta * m[name]
                                param.data = param.data - gx_lr * m[name]

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

                        print(f'epoch{epoch+1} validation set: loss={float(ave_loss.item()):.6f}, '
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

        # """  第一次相同超参数下不同lr的对比，以确定更准确的lr搜索范围 """
        # colors = matplotlib.colormaps.get_cmap('tab10')  # 颜色映射
        # T = 20
        # plt.figure(figsize=(10, 6))
        # for index in range(len(lr_list)):
        #     file = f'../save_data/{file_name}_{model_name}_{dataset_name}/ave_{optimizer}_{parameters}_{lr_list[index]}.pkl'
        #     with open(file, 'rb') as f:
        #         data = pickle.load(f)
        #     train_loss = np.array(data['train_loss_list'])
        #
        #     x_values = list(range(T, len(train_loss)))
        #     plt.plot(x_values, train_loss[T:], label=f'{optimizer} parameters{parameters} lr{lr_list[index]}',
        #              color=colors(index))
        #
        # # plt.plot(np.full(Epoch, 0), label='0')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.grid()
        # plt.savefig(f'../save_tu/{file_name}_{model_name}_{dataset_name}/loss/{optimizer}_{parameters}_loss.png')
        # plt.clf()  # 清除图像内容
        # plt.close()  # 关闭图像
        #
        # plt.figure(figsize=(10, 6))
        # for index in range(len(lr_list)):
        #     file = f'../save_data/{file_name}_{model_name}_{dataset_name}/ave_{optimizer}_{parameters}_{lr_list[index]}.pkl'
        #     with open(file, 'rb') as f:
        #         data = pickle.load(f)
        #     validation_acc = np.array(data['validation_acc_list'])
        #
        #     x_values = list(range(T, len(validation_acc)))
        #     plt.plot(x_values, validation_acc[T:], label=f'{optimizer} parameters{parameters} lr{lr_list[index]}',
        #              linestyle='--', color=colors(index))
        # # plt.plot(np.full(Epoch, 1), label='1')
        # plt.xlabel('Epoch')
        # plt.ylabel('ACC')
        # plt.legend()
        # plt.grid()
        # plt.savefig(f'../save_tu/{file_name}_{model_name}_{dataset_name}/acc/{optimizer}_{parameters}_acc.png')
        # plt.clf()  # 清除图像内容
        # plt.close()  # 关闭图像
        #
        # print("tu saved successfully.")

