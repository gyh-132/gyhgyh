import torch
import torch.nn as nn
import numpy as np
import Model
import pickle
import matplotlib.pyplot as plt
import matplotlib
from data import dataloader

# 实验需求超参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device= ', device)

L2 = ['t', 1e-4]      # L2正则    分别表示：是否启用、正则项系数
yr = ['t', 10, 0.1]   # 学习率预热 分别表示：是否启用、预热周期、初始学习率与固定学习率的比例
TH = ['t', 30, 0.2]     # 学习率退火 分别表示：是否启用、执行退火的间隔、退火系数

file_name = 'AdaQHN'
model_name, dataset_name = 'GLeNet', 'FMnist'  # 模型、数据集
Epoch, batch_size = 80, 128  # 周期、样本分批大小
# opt_list = ['Adam', 'QHAdam', 'Adan', 'Adam_win', 'AdaQHN']  # 优化算法
# lr_list = [0.001, 0.0015, 0.002, 0.003, 0.005, 0.007, 0.01]  # 学习率
# # seed_list = torch.randint(10, 1000, (3, 2)).numpy()
# # print('seed_list:', seed_list)
# seed_list = [[825, 831], [253, 995], [192, 418], [629, 808], [390, 427]]  # 模型参数初始化与数据集分批所需随机种子
seed_list = [[825, 831], [253, 995], [192, 418]]  # 模型参数初始化与数据集分批所需随机种子
lr_list = [0.001, 0.0015, 0.002, 0.003, 0.005, 0.007, 0.01]  # 学习率
opt_list = ['Adam', 'QHAdam', 'Adan', 'Adam_win', 'AdaQHN']  # 优化算法

# 开始实验
for opt in range(len(opt_list)):
    optimizer = opt_list[opt]

    if optimizer == 'Adam':
        beta1, beta2 = 0.9, 0.999
    elif optimizer == 'Adam_win':
        beta1, beta2, wd = 0.9, 0.999, 1e-4  # wd为权重衰减系数 weight decay
    elif optimizer == 'Adan':
        beta1, beta2, beta3, wd = 0.98, 0.92, 0.99, 1e-4
    elif optimizer == 'QHAdam':
        beta1, beta2, v = 0.999, 0.999, 0.7
    elif optimizer == 'AdaQHN':
        beta1, beta2, v = 0.999, 0.999, 0.8
    else:
        print('optimizer chu cuo!')

    for lr_index in range(len(lr_list)):
        lr = lr_list[lr_index]

        for seed_index in range(len(seed_list)):
            model_seed = seed_list[seed_index][0]
            data_seed = seed_list[seed_index][1]

            print('model:{}, dataset:{}, optimizer:{}'.format(model_name, dataset_name, optimizer))
            print('lr:{}, model_seed:{}, data_seed:{}'.format(lr, model_seed, data_seed))

            # 导入数据集
            train_loader, validation_loader = dataloader(data_seed, batch_size, dataset_name)
            # 定义模型、损失函数
            if model_name == 'GLeNet':
                model = Model.GLeNet(model_seed).to(device)
                criterion = nn.CrossEntropyLoss()
            else:
                print("model cuo wu!")

            # 储存实验数据，记录每一epoch后模型在训练集和验证集上的损失与准确率
            train_loss_list = []
            train_acc_list = []
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
            lin_w = {}  # 记录额外的模型参数(AdaQHN的外推梯度，win中第一个参数序列)
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
                            h[name] = (1 - beta3) * ((param.grad + beta2*(param.grad-g[name])) ** 2) + beta3 * h[name]
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
                            # 更新自适应步长项并修正当前外推梯度
                            h[name] = (1 - beta2) * (param.grad ** 2) + beta2 * h[name]
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
                        print('Epoch: {}, iter: {}, loss: {:.6f}'.format(epoch+1, i+1, float(loss.item())))
                # 模型设置为评估模式
                model.eval()
                with torch.no_grad():
                    # 评估训练集
                    total_loss = 0
                    total_correct = 0
                    total_samples = 0

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

                    print('epoch{} train set: loss={:.6f}，Accuracy={:.2f}%'.format(epoch + 1, float(ave_loss.item()),
                                                                                   accuracy * 100))

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

                    print('epoch{} validation set: loss={:.6f}，Accuracy={:.2f}%'.format(epoch + 1, float(ave_loss.item()),
                                                                                        accuracy * 100))

            # 保存实验数据
            file = '../save_data/{}_{}_{}/{}_lr{}_seed{}.pkl' \
                .format(file_name, model_name, dataset_name, optimizer, lr, seed_index)
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
            file = '../save_data/{}_{}_{}/{}_lr{}_seed{}.pkl'\
                .format(file_name, model_name, dataset_name, optimizer, lr, p)
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

        file = '../save_data/{}_{}_{}/ave_{}_lr{}.pkl'.format(file_name, model_name, dataset_name, optimizer, lr)
        with open(file, 'wb') as f:
            pickle.dump({
                'train_loss_list': ave_train_loss,
                'train_acc_list': ave_train_acc,
                'validation_loss_list': ave_validation_loss,
                'validation_acc_list': ave_validation_acc,
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
        train_acc = np.array(data['train_acc_list'])
        validation_loss = np.array(data['validation_loss_list'])
        validation_acc = np.array(data['validation_acc_list'])

        plt.plot(train_loss, label='{} lr{} train'
                 .format(optimizer, lr_list[index]), color=colors(index))
        plt.plot(validation_loss, label='{} lr{} validation'
                 .format(optimizer, lr_list[index]), linestyle='--', color=colors(index))
    plt.plot(np.full(Epoch, 0), label='0')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig('../save_tu/{}_{}_{}/{}_loss.png'.format(file_name, model_name, dataset_name, optimizer))
    plt.clf()  # 清除图像内容
    plt.close()  # 关闭图像

    plt.figure(figsize=(10, 6))
    for index in range(len(lr_list)):
        file = '../save_data/{}_{}_{}/ave_{}_lr{}.pkl'\
            .format(file_name, model_name, dataset_name, optimizer, lr_list[index])
        with open(file, 'rb') as f:
            data = pickle.load(f)
        train_loss = np.array(data['train_loss_list'])
        train_acc = np.array(data['train_acc_list'])
        validation_loss = np.array(data['validation_loss_list'])
        validation_acc = np.array(data['validation_acc_list'])

        plt.plot(train_acc, label='{} lr{} train'
                 .format(optimizer, lr_list[index]), color=colors(index))
        plt.plot(validation_acc, label='{} lr{} validation'
                 .format(optimizer, lr_list[index]), linestyle='--', color=colors(index))
    plt.plot(np.full(Epoch, 1), label='1')
    plt.xlabel('Epoch')
    plt.ylabel('ACC')
    plt.legend()
    plt.grid()
    plt.savefig('../save_tu/{}_{}_{}/{}_acc.png'.format(file_name, model_name, dataset_name, optimizer))
    plt.clf()  # 清除图像内容
    plt.close()  # 关闭图像

    print("tu saved successfully.")
