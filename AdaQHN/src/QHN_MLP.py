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
from untils import RMSELoss

# 导入超参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device= ', device)
# 实验需求超参数
file_name = 'QHN'
model_name, dataset_name, optimizer = 'MLP1', 'DXS1', 'SGD'  # 模型、数据集与优化器
model_seed, data_seed = 42, 42  # 模型参数初始化与数据集分批所需随机种子
lr, Epoch, batch_size = 1, 150, 128  # 学习率、周期、样本分批大小

# 按需求修改
# seed_list = torch.randint(10, 1000, (3, 2)).numpy()
# print('seed_list:', seed_list)
# seed_list = [[825, 831], [253, 995], [192, 418], [629, 808], [390, 427]]
# bs_list = [64, 128, 256]
# opt_list = ['SGD', 'momentum', 'NAG', 'RNAG', 'QHM', 'QHN']
# lr_list = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  # SGD
# beta_list = [0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995]  # Momentum、NAG、RNAG
# beta_list = [0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995]  # QHM、QHN
# v_list = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999]

seed_list = [[825, 831], [253, 995], [192, 418]]
opt_list = ['QHM', 'QHN']
lr_list = [0.5, 0.7, 1.2, 1.5]

L2 = ['t', 1e-4]      # L2正则    分别表·示：是否启用、正则项系数
yr = ['t', 10, 0.1]   # 学习率预热 分别表示：是否启用、预热周期、初始学习率与固定学习率的比例
TH = ['t', 60, 0.2]     # 学习率退火 分别表示：是否启用、执行退火的间隔、退火系数

for lr_index in range(len(lr_list)):
    lr = lr_list[lr_index]

    for opt in range(len(opt_list)):
        optimizer = opt_list[opt]

        if optimizer == 'SGD':
            beta_list = [0.9]
            v_list = [1]
        elif optimizer == 'momentum':
            beta_list = [0.95]
            v_list = [1]
        elif optimizer == 'NAG':
            beta_list = [0.95]
            v_list = [1]
        elif optimizer == 'RNAG':
            beta_list = [0.95]
            v_list = [1]
        elif optimizer == 'QHM':
            beta_list = [0.99]
            v_list = [0.95]
        elif optimizer == 'QHN':
            beta_list = [0.99]
            v_list = [0.95]
        else:
            print('optimizer chu cuo!')

        for beta_index in range(len(beta_list)):
            beta = beta_list[beta_index]

            for v_index in range(len(v_list)):
                v = v_list[v_index]

                if optimizer != 'SGD':
                    SGD_file = f'../save_data/{file_name}_{model_name}_{dataset_name}/ave_SGD_beta0.9_v1_lr1.pkl'
                    with open(SGD_file, 'rb') as f:
                        SGD_data = pickle.load(f)
                    SGD_train_loss = SGD_data['train_loss_list']
                    SGD_validation_loss = SGD_data['validation_loss_list']

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
                    if model_name == 'MLP1':
                        model = Model.MLP1(model_seed).to(device)
                        criterion = RMSELoss()
                    else:
                        print("model cuo wu!")

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
                                loss = criterion(output, y)  # 求损失
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

                        # 模型设置为评估模式
                        model.eval()
                        with torch.no_grad():
                            # 评估训练集
                            total_loss = 0
                            for j, (x, y) in enumerate(train_loader):
                                # 将输入数据移动到GPU上
                                x, y = x.to(device), y.to(device)
                                output = model(x)
                                loss = criterion(output, y)
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
                                loss = criterion(output, y)
                                total_loss += loss

                            ave_loss = total_loss / len(validation_loader)
                            validation_loss_list.append(float(ave_loss.item()))

                            print('epoch{} loss{} on the test set'.format(epoch + 1, float(ave_loss.item())))

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

                file = '../save_data/{}_{}_{}/ave_{}_beta{}_v{}_lr{}.pkl' \
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
                             label='{} β{} v{} train'.format(optimizer, beta, v))
                    plt.plot(ave_validation_loss,
                             label='{} β{} v{} validation'.format(optimizer, beta, v))
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.grid()
                    plt.savefig(
                        '../save_tu/{}_{}_{}/{}_beta{}_v{}_lr{}_loss.png'
                        .format(file_name, model_name, dataset_name, optimizer, beta, v, lr))
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
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.grid()
                    plt.savefig(
                        '../save_tu/{}_{}_{}/{}_beta{}_v{}_lr{}_loss.png'
                        .format(file_name, model_name, dataset_name, optimizer, beta, v, lr))
                    plt.clf()  # 清除图像内容
                    plt.close()  # 关闭图像

                print("tu saved successfully.")

