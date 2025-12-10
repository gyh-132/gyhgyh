import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.patches import ConnectionPatch
import itertools

# ################################  QHN 为各算法选择最优参数，按最低loss和最高acc  ##############################################
"""
QHN

各算法搜索范围分别为momentum,NAG[0.8, 0.85, 0.9, 0.95, 0.99]; 
QHM,QHN[[0.8, 0.85, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995], [0.5, 0.6, 0.7, 0.8, 0.9]]

在['PoolMLP', 'EMnist_digits']上，第二次lr搜索范围分别为SGD[0.5, 0.7, 1, 1.2, 1.5]，其它[2, 2.5, 3, 3.5, 4]

在['TinyVGG', 'EMnist_letters']上，第二次lr搜索范围分别为SGD[0.15, 0.2, 0.3, 0.5, 0.7]，其它[0.7, 1, 1.2, 1.5, 2]

在['MobileCNN', 'EMnist_balanced']上，第二次lr搜索范围分别为SGD[0.2, 0.3, 0.5, 0.7, 1];其它[0.7, 1, 1.2, 1.5, 2]
"""

# file_name = 'QHN'
# model_name, dataset_name, = 'MobileCNN', 'EMnist_balanced'  # 模型、数据集
# # opt_list = ['SGD', 'momentum', 'NAG', 'QHM', 'QHN']
# opt_list = ['SGD', 'momentum', 'NAG', 'QHM', 'QHN']
#
# save_dir = f'../save_tu/'
# os.makedirs(save_dir, exist_ok=True)
# with open(save_dir + f"{file_name}_{model_name}_{dataset_name}_select_params.txt", "w") as file:  # 先清空文件
#     pass  # 只是确保文件存在且为空
#
# window_size = 10  # 使用最后window_size个数据点计算均值
# loss_start_epoch = 20  # 从第 loss_start_epoch 个数据点开始绘制loss图
# acc_start_epoch = 20   # 从第 acc_start_epoch 个数据点开始绘制acc图
# loss_limit = ['1', 0.03]   # 首元素为T表示绘制等值线
# acc_limit = ['1', 0.92]
# decimal_places = 5  # 设置保留的小数位数，可以根据需要调整
# fix_params = 'T'  # 选择T表示固定算法超参数，仅对学习率进行搜索，用来查看各算法默认参数下的表现情况
#
# # 用于存储绘图数据的字典
# plot_data = {
#     'loss': {'values': [], 'labels': [], 'files': []},
#     'acc': {'values': [], 'labels': [], 'files': []}
# }
#
# for opt_index in range(len(opt_list)):
#     optimizer = opt_list[opt_index]
#
#     # 每次循环时重置最小损失和最大准确率
#     min_loss = 1e4
#     max_acc = 0
#     loss_params_file = ''
#     acc_params_file = ''
#
#     if optimizer == 'SGD':
#         beta_list = [0]
#         parameters_list = [beta_list]
#         lr_list = [0.2, 0.3, 0.5, 0.7, 1]
#     elif optimizer == 'momentum':
#         if fix_params == 'T':
#             beta_list = [0.9]
#         else:
#             beta_list = [0.8, 0.85, 0.9, 0.95, 0.99]
#         parameters_list = [beta_list]
#         lr_list = [0.7, 1, 1.2, 1.5, 2]
#     elif optimizer == 'NAG':
#         if fix_params == 'T':
#             beta_list = [0.9]
#         else:
#             beta_list = [0.8, 0.85, 0.9, 0.95, 0.99]
#         parameters_list = [beta_list]
#         lr_list = [0.7, 1, 1.2, 1.5, 2]
#     elif optimizer == 'QHM':
#         if fix_params == 'T':
#             beta_list = [0.95]
#             v_list = [0.7]
#         else:
#             beta_list = [0.95, 0.99, 0.995, 0.999, 0.9995]
#             v_list = [0.5, 0.6, 0.7, 0.8, 0.9]
#         parameters_list = [beta_list, v_list]
#         lr_list = [0.7, 1, 1.2]
#     elif optimizer == 'QHN':
#         if fix_params == 'T':
#             beta_list = [0.95]
#             v_list = [0.7]
#         else:
#             beta_list = [0.95, 0.99, 0.995, 0.999, 0.9995]
#             v_list = [0.5, 0.6, 0.7, 0.8, 0.9]
#         parameters_list = [beta_list, v_list]
#         lr_list = [0.7, 1, 1.2, 1.5, 2]
#     else:
#         raise ValueError(f"Unknown optimizer: {optimizer}")
#
#     for parameters in itertools.product(*parameters_list):
#         if optimizer in ['SGD', 'momentum', 'NAG']:
#             beta = parameters[0]
#         elif optimizer in ['QHM', 'QHN']:
#             beta, v = parameters
#         else:
#             raise ValueError(f"Invalid parameter set for optimizer: {optimizer}")
#
#         for lr in lr_list:
#             file = f'../save_data/{file_name}_{model_name}_{dataset_name}/ave_{optimizer}_{parameters}_{lr}.pkl'
#             try:
#                 with open(file, 'rb') as f:
#                     data = pickle.load(f)
#                 train_loss_list = np.array(data['train_loss_list'])
#                 validation_acc_list = np.array(data['validation_acc_list'])
#
#                 min_lin_loss = np.mean(train_loss_list[-window_size:])
#                 max_lin_acc = np.mean(validation_acc_list[-window_size:])
#
#                 # 更新最小损失
#                 if min_lin_loss < min_loss:
#                     min_loss = min_lin_loss
#                     loss_params_file = file
#                     best_loss_data = data  # 保存最小loss对应的数据
#
#                 # 更新最大准确率
#                 if max_lin_acc > max_acc:
#                     max_acc = max_lin_acc
#                     acc_params_file = file
#                     best_acc_data = data  # 保存最大acc对应的数据
#             except FileNotFoundError:
#                 print(f"File not found: {file}")
#                 continue
#
#     if loss_params_file and acc_params_file:
#         # 格式化数值，保留指定小数位数
#         formatted_min_loss = round(min_loss, decimal_places)
#         formatted_max_acc = round(max_acc, decimal_places)
#
#         with open(save_dir + f"{file_name}_{model_name}_{dataset_name}_select_params.txt", "a") as file:  # 注意改为 "a" 模式
#             file.write(f"{optimizer} loss_file: {loss_params_file} loss: {formatted_min_loss}\n")
#             file.write(f"{optimizer} acc_file: {acc_params_file} acc: {formatted_max_acc}\n")
#
#         # 存储绘图数据，并在标签中添加min_loss和max_acc
#         if optimizer == 'momentum':
#             loss_label = f"CM (loss={formatted_min_loss})"
#             acc_label = f"CM (acc={formatted_max_acc})"
#         else:
#             loss_label = f"{optimizer} (loss={formatted_min_loss})"
#             acc_label = f"{optimizer} (acc={formatted_max_acc})"
#
#         plot_data['loss']['values'].append(best_loss_data['train_loss_list'][loss_start_epoch:])
#         plot_data['loss']['labels'].append(loss_label)
#         plot_data['loss']['files'].append(loss_params_file)
#
#         plot_data['acc']['values'].append(best_acc_data['validation_acc_list'][acc_start_epoch:])
#         plot_data['acc']['labels'].append(acc_label)
#         plot_data['acc']['files'].append(acc_params_file)
#
# # 绘制损失对比图
# plt.figure(figsize=(8, 6))
# for i, (values, label) in enumerate(zip(plot_data['loss']['values'], plot_data['loss']['labels'])):
#     plt.plot(range(loss_start_epoch, loss_start_epoch + len(values)), values, label=label, linewidth=2)
# if loss_limit[0] == 'T':
#     plt.plot(range(loss_start_epoch, loss_start_epoch + len(values)), np.full(len(values), loss_limit[1]))
# plt.title(f'Training Loss Comparison (From Epoch {loss_start_epoch}, loss=ave[-{window_size}:])', fontsize=16)
# plt.xlabel('Epoch', fontsize=16)
# plt.ylabel('Loss', fontsize=16)
# plt.legend(fontsize=18)  # 设置图例字体大小
# plt.grid(True)
# plt.tight_layout()
# if fix_params == 'T':
#     loss_plot_path = f'{save_dir}{file_name}_{model_name}_{dataset_name}_fix_loss.png'
# else:
#     loss_plot_path = f'{save_dir}{file_name}_{model_name}_{dataset_name}_loss.png'
# plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
# plt.close()
# print(f"损失对比图已保存到: {loss_plot_path}")
#
# # 绘制准确率对比图
# plt.figure(figsize=(8, 6))
# for i, (values, label) in enumerate(zip(plot_data['acc']['values'], plot_data['acc']['labels'])):
#     plt.plot(range(acc_start_epoch, acc_start_epoch + len(values)), values, label=label, linewidth=2)
# if acc_limit[0] == 'T':
#     plt.plot(range(acc_start_epoch, acc_start_epoch + len(values)), np.full(len(values), acc_limit[1]))
# plt.title(f'Validation Accuracy Comparison (From Epoch {acc_start_epoch}, acc=ave[-{window_size}:])', fontsize=16)
# plt.xlabel('Epoch', fontsize=16)
# plt.ylabel('Accuracy', fontsize=16)
# plt.legend(fontsize=18)  # 设置图例字体大小
# plt.grid(True)
# plt.tight_layout()
# if fix_params == 'T':
#     acc_plot_path = f'{save_dir}{file_name}_{model_name}_{dataset_name}_fix_acc.png'
# else:
#     acc_plot_path = f'{save_dir}{file_name}_{model_name}_{dataset_name}_acc.png'
# plt.savefig(acc_plot_path, dpi=300, bbox_inches='tight')
# plt.close()
# print(f"准确率对比图已保存到: {acc_plot_path}")

# ################################   QHN 不同算法在特殊参数选择下对比     ###################################
"""
QHN

各算法超参数分别为momentum,NAG[0.9]; QHM,QHN[[0.95, 0.999], 0.7]
学习率由上面画图程序在固定参数选择下(fix_params = 'T')确定

在['PoolMLP', 'EMnist_digits']上, lr分别为[1.5, 4, 4, 4, 3.5, 4, 4]
在['TinyVGG', 'EMnist_letters']上, lr分别为[0.7, 1.5, 1.5, 2, 1.5, 2, 2]
在['MobileCNN', 'EMnist_balanced']上, lr分别为[0.7, 1.5, 2, 1.2, 2, 1.5, 2]

"""

# model_name, dataset_name, optimizer= 'MobileCNN', 'EMnist_balanced', 'QHN'  # 模型, 数据集, 算法
# lr_list = [0.7, 1.5, 2, 1.2, 2, 1.5, 2]
# window_size = 10  # 使用最后window_size个数据点计算均值
# loss_start_epoch = 10  # 从第 loss_start_epoch 个数据点开始绘制loss图
# decimal_places = 5  # 设置保留的小数位数，可以根据需要调整
#
# plot_data = {'values': [], 'labels': []}
#
# SGD_file = f'../save_data/{optimizer}_{model_name}_{dataset_name}/ave_SGD_(0,)_{lr_list[0]}.pkl'
# with open(SGD_file, 'rb') as f:
#     SGD_data = pickle.load(f)
# SGD_train_loss_list = np.array(SGD_data['train_loss_list'])
# SGD_min_loss = round(np.mean(SGD_train_loss_list[-window_size:]), decimal_places)
# plot_data['values'].append(SGD_train_loss_list[loss_start_epoch:])
# plot_data['labels'].append(f'SGD, loss={SGD_min_loss}')
#
# CM_file = f'../save_data/{optimizer}_{model_name}_{dataset_name}/ave_momentum_(0.9,)_{lr_list[1]}.pkl'
# with open(CM_file, 'rb') as f:
#     CM_data = pickle.load(f)
# CM_train_loss_list = np.array(CM_data['train_loss_list'])
# CM_min_loss = round(np.mean(CM_train_loss_list[-window_size:]), decimal_places)
# plot_data['values'].append(CM_train_loss_list[loss_start_epoch:])
# plot_data['labels'].append(f'CM(β=0.9), loss={CM_min_loss}')
#
# NAG_file = f'../save_data/{optimizer}_{model_name}_{dataset_name}/ave_NAG_(0.9,)_{lr_list[2]}.pkl'
# with open(NAG_file, 'rb') as f:
#     NAG_data = pickle.load(f)
# NAG_train_loss_list = np.array(NAG_data['train_loss_list'])
# NAG_min_loss = round(np.mean(NAG_train_loss_list[-window_size:]), decimal_places)
# plot_data['values'].append(NAG_train_loss_list[loss_start_epoch:])
# plot_data['labels'].append(f'NAG(β=0.9), loss={NAG_min_loss}')
#
# QHM_file1 = f'../save_data/{optimizer}_{model_name}_{dataset_name}/ave_QHM_(0.95, 0.7)_{lr_list[3]}.pkl'
# with open(QHM_file1, 'rb') as f:
#     QHM_data1 = pickle.load(f)
# QHM_train_loss_list1 = np.array(QHM_data1['train_loss_list'])
# QHM_min_loss1 = round(np.mean(QHM_train_loss_list1[-window_size:]), decimal_places)
# plot_data['values'].append(QHM_train_loss_list1[loss_start_epoch:])
# plot_data['labels'].append(f'QHM(β=0.95, v=0.7), loss={QHM_min_loss1}')
#
# QHM_file2 = f'../save_data/{optimizer}_{model_name}_{dataset_name}/ave_QHM_(0.999, 0.7)_{lr_list[4]}.pkl'
# with open(QHM_file2, 'rb') as f:
#     QHM_data2 = pickle.load(f)
# QHM_train_loss_list2 = np.array(QHM_data2['train_loss_list'])
# QHM_min_loss2 = round(np.mean(QHM_train_loss_list2[-window_size:]), decimal_places)
# plot_data['values'].append(QHM_train_loss_list2[loss_start_epoch:])
# plot_data['labels'].append(f'QHM(β=0.999, v=0.7), loss={QHM_min_loss2}')
#
# QHN_file1 = f'../save_data/{optimizer}_{model_name}_{dataset_name}/ave_QHN_(0.95, 0.7)_{lr_list[5]}.pkl'
# with open(QHN_file1, 'rb') as f:
#     QHN_data1 = pickle.load(f)
# QHN_train_loss_list1 = np.array(QHN_data1['train_loss_list'])
# QHN_min_loss1 = round(np.mean(QHN_train_loss_list1[-window_size:]), decimal_places)
# plot_data['values'].append(QHN_train_loss_list1[loss_start_epoch:])
# plot_data['labels'].append(f'QHN(β=0.95, v=0.7), loss={QHN_min_loss1}')
#
# QHN_file2 = f'../save_data/{optimizer}_{model_name}_{dataset_name}/ave_QHN_(0.999, 0.7)_{lr_list[6]}.pkl'
# with open(QHN_file2, 'rb') as f:
#     QHN_data2 = pickle.load(f)
# QHN_train_loss_list2 = np.array(QHN_data2['train_loss_list'])
# QHN_min_loss2 = round(np.mean(QHN_train_loss_list2[-window_size:]), decimal_places)
# plot_data['values'].append(QHN_train_loss_list2[loss_start_epoch:])
# plot_data['labels'].append(f'QHN(β=0.999, v=0.7), loss={QHN_min_loss2}')
#
# # 绘制损失对比图
# plt.figure(figsize=(8, 6))
# for i, (values, label) in enumerate(zip(plot_data['values'], plot_data['labels'])):
#     plt.plot(range(loss_start_epoch, loss_start_epoch + len(values)), values, label=label, linewidth=2)
# plt.title(f'Training Loss Comparison (From Epoch {loss_start_epoch}, loss=ave[-{window_size}:])', fontsize=16)
# plt.xlabel('Epoch', fontsize=16)
# plt.ylabel('Training Loss', fontsize=16)
# plt.legend(fontsize=16)  # 设置图例字体大小
# plt.grid(True)
# plt.tight_layout()
# plot_path = f'../save_tu/{optimizer}_{model_name}_{dataset_name}_fix_loss.png'
# plt.savefig(plot_path, dpi=300, bbox_inches='tight')
# plt.close()
# print(f"图已保存到: {plot_path}")

# ################################   AdaQHN 为各算法选择最优参数，按最低loss和最高acc     ###################################
"""
AdaQHN

各算法超参数搜索范围分别为Adam[[0.8, 0.85, 0.9, 0.95, 0.99], 0.999]; Adam_win[[0.8, 0.85, 0.9, 0.95, 0.99], 0.999];
Adan[[0.9, 0.93, 0.96, 0.99], [0.9, 0.93, 0.96, 0.99], [0.9, 0.93, 0.96, 0.99]]
QHAdam[[0.95, 0.99, 0.995, 0.999, 0.9995], 0.999, [0.5, 0.6, 0.7, 0.8, 0.9]];
AdaQHN[[0.95, 0.99, 0.995, 0.999, 0.9995], 0.999, [0.5, 0.6, 0.7, 0.8, 0.9]]

在['PoolMLP', 'EMnist_digits']，第二次lr搜索范围为  Adam、Adam_win[0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05];
和Adan、QHAdam、AdaQHN[0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07]

在['GLeNet5', 'FMnist']上，第二次lr搜索范围均为[0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03];

在['TinyVGG', 'EMnist_letters']上，第二次lr搜索范围均为[0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05]

在['MicroVGG', 'cifar10']上，第二次lr搜索范围分别为 Adam, Adam_win, Adan[0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05];
QHAdam和AdaQHN[0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07]

在['GRU1', 'AGnews']上，第二次lr搜索范围均为[0.0007, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.007]

在['LSTM1', 'AGnews']上，第二次lr搜索范围分别为 
Adam、Adam_win[0.0001, 0.00015, 0.0002, 0.0003, 0.0005, 0.0007, 0.001];
Adan[0.0005, 0.0007, 0.001, 0.0015, 0.002, 0.003, 0.005];
QHMAdam和AdaQHN[0.001, 0.0015, 0.002, 0.003, 0.005, 0.007, 0.01]
"""

# file_name = 'AdaQHN'
# model_name, dataset_name, = 'LSTM1', 'AGnews'  # 模型、数据集
# opt_list = ['Adam', 'Adam_win', 'Adan', 'QHAdam', 'AdaQHN']
#
# save_dir = f'../save_tu/'
# os.makedirs(save_dir, exist_ok=True)
# with open(save_dir + f"{file_name}_{model_name}_{dataset_name}_select_params.txt", "w") as file:  # 先清空文件
#     pass  # 只是确保文件存在且为空
#
# window_size = 10  # 使用最后window_size个数据点计算均值
# loss_start_epoch = 10  # 从第 loss_start_epoch 个数据点开始绘制loss图
# acc_start_epoch = 10  # 从第 acc_start_epoch 个数据点开始绘制acc图
# loss_limit = ['1', 0.03]  # T
# acc_limit = ['1', 0.92]
# decimal_places = 5  # 设置保留的小数位数，可以根据需要调整
# fix_params = 'T'
#
# # 用于存储绘图数据的字典
# plot_data = {
#     'loss': {'values': [], 'labels': [], 'files': []},
#     'acc': {'values': [], 'labels': [], 'files': []}
# }
#
# for opt_index in range(len(opt_list)):
#     optimizer = opt_list[opt_index]
#
#     # 每次循环时重置最小损失和最大准确率
#     min_loss = 1e4
#     max_acc = 0
#     loss_params_file = ''
#     acc_params_file = ''
#
#     if optimizer == 'Adam':
#         if fix_params == 'T':
#             beta1_list = [0.9]
#         else:
#             beta1_list = [0.8, 0.85, 0.9, 0.95, 0.99]
#         beta2_list = [0.999]
#         parameters_list = [beta1_list, beta2_list]
#         lr_list = [0.0001, 0.00015, 0.0002, 0.0003, 0.0005, 0.0007, 0.001]
#     elif optimizer == 'Adam_win':
#         if fix_params == 'T':
#             beta1_list = [0.9]
#         else:
#             beta1_list = [0.8, 0.85, 0.9, 0.95, 0.99]
#         beta2_list = [0.999]
#         parameters_list = [beta1_list, beta2_list]
#         lr_list = [0.0001, 0.00015, 0.0002, 0.0003, 0.0005, 0.0007, 0.001]
#     elif optimizer == 'Adan':
#         if fix_params == 'T':
#             beta1_list = [0.96]
#             beta2_list = [0.9]
#             beta3_list = [0.99]
#         else:
#             beta1_list = [0.9, 0.93, 0.96, 0.99]
#             beta2_list = [0.9, 0.93, 0.96, 0.99]
#             beta3_list = [0.9, 0.93, 0.96, 0.99]
#         parameters_list = [beta1_list, beta2_list, beta3_list]
#         lr_list = [0.0005, 0.0007, 0.001, 0.0015, 0.002, 0.003, 0.005]
#     elif optimizer == 'QHAdam':
#         if fix_params == 'T':
#             beta1_list = [0.999]
#             v_list = [0.7]
#         else:
#             beta1_list = [0.95, 0.99, 0.995, 0.999, 0.9995]
#             v_list = [0.5, 0.6, 0.7, 0.8, 0.9]
#         beta2_list = [0.999]
#         parameters_list = [beta1_list, beta2_list, v_list]
#         lr_list = [0.001, 0.0015, 0.002, 0.003, 0.005, 0.007, 0.01]
#     elif optimizer == 'AdaQHN':
#         if fix_params == 'T':
#             beta1_list = [0.999]
#             v_list = [0.7]
#         else:
#             beta1_list = [0.95, 0.99, 0.995, 0.999, 0.9995]
#             v_list = [0.5, 0.6, 0.7, 0.8, 0.9]
#         beta2_list = [0.999]
#         parameters_list = [beta1_list, beta2_list, v_list]
#         lr_list = [0.001, 0.0015, 0.002, 0.003, 0.005, 0.007, 0.01]
#     else:
#         raise ValueError(f"Unknown optimizer: {optimizer}")
#
#     for parameters in itertools.product(*parameters_list):
#         if optimizer in ['Adam', 'Adam_win']:
#             beta1, beta2 = parameters
#         elif optimizer == 'Adan':
#             beta1, beta2, beta3 = parameters
#         elif optimizer in ['QHAdam', 'AdaQHN']:
#             beta1, beta2, v = parameters
#         else:
#             raise ValueError(f"Invalid parameter set for optimizer: {optimizer}")
#
#         for lr in lr_list:
#             file = f'../save_data/{file_name}_{model_name}_{dataset_name}/ave_{optimizer}_{parameters}_{lr}.pkl'
#             try:
#                 with open(file, 'rb') as f:
#                     data = pickle.load(f)
#                 train_loss_list = np.array(data['train_loss_list'])
#                 validation_acc_list = np.array(data['validation_acc_list'])
#
#                 min_lin_loss = np.mean(train_loss_list[-window_size:])
#                 max_lin_acc = np.mean(validation_acc_list[-window_size:])
#
#                 # 更新最小损失
#                 if min_lin_loss < min_loss:
#                     min_loss = min_lin_loss
#                     loss_params_file = file
#                     best_loss_data = data  # 保存最小loss对应的数据
#
#                 # 更新最大准确率
#                 if max_lin_acc > max_acc:
#                     max_acc = max_lin_acc
#                     acc_params_file = file
#                     best_acc_data = data  # 保存最大acc对应的数据
#             except FileNotFoundError:
#                 print(f"File not found: {file}")
#                 continue
#
#     if loss_params_file and acc_params_file:
#         # 格式化数值，保留指定小数位数
#         formatted_min_loss = round(min_loss, decimal_places)
#         formatted_max_acc = round(max_acc, decimal_places)
#
#         with open(save_dir + f"{file_name}_{model_name}_{dataset_name}_select_params.txt", "a") as file:  # 注意改为 "a" 模式
#             file.write(f"{optimizer} loss_file: {loss_params_file} loss: {formatted_min_loss}\n")
#             file.write(f"{optimizer} acc_file: {acc_params_file} acc: {formatted_max_acc}\n")
#
#         # 存储绘图数据，并在标签中添加min_loss和max_acc
#         loss_label = f"{optimizer} (loss={formatted_min_loss})"
#         acc_label = f"{optimizer} (acc={formatted_max_acc})"
#
#         plot_data['loss']['values'].append(best_loss_data['train_loss_list'][loss_start_epoch:])
#         plot_data['loss']['labels'].append(loss_label)
#         plot_data['loss']['files'].append(loss_params_file)
#
#         plot_data['acc']['values'].append(best_acc_data['validation_acc_list'][acc_start_epoch:])
#         plot_data['acc']['labels'].append(acc_label)
#         plot_data['acc']['files'].append(acc_params_file)
#
# # 绘制损失对比图
# plt.figure(figsize=(8, 6))
# for i, (values, label) in enumerate(zip(plot_data['loss']['values'], plot_data['loss']['labels'])):
#     plt.plot(range(loss_start_epoch, loss_start_epoch + len(values)), values, label=label, linewidth=2)
# if loss_limit[0] == 'T':
#     plt.plot(range(loss_start_epoch, loss_start_epoch + len(values)), np.full(len(values), loss_limit[1]))
# plt.title(f'Training Loss Comparison (From Epoch {loss_start_epoch}, loss=ave[-{window_size}:])', fontsize=16)
# plt.xlabel('Epoch', fontsize=16)
# plt.ylabel('Loss', fontsize=16)
# plt.legend(fontsize=18)  # 设置图例字体大小
# plt.grid(True)
# plt.tight_layout()
# if fix_params == 'T':
#     loss_plot_path = f'{save_dir}{file_name}_{model_name}_{dataset_name}_fix_loss.png'
# else:
#     loss_plot_path = f'{save_dir}{file_name}_{model_name}_{dataset_name}_loss.png'
# plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
# plt.close()
# print(f"损失对比图已保存到: {loss_plot_path}")
#
# # 绘制准确率对比图
# plt.figure(figsize=(8, 6))
# for i, (values, label) in enumerate(zip(plot_data['acc']['values'], plot_data['acc']['labels'])):
#     plt.plot(range(acc_start_epoch, acc_start_epoch + len(values)), values, label=label, linewidth=2)
# if acc_limit[0] == 'T':
#     plt.plot(range(acc_start_epoch, acc_start_epoch + len(values)), np.full(len(values), acc_limit[1]))
# plt.title(f'Validation Accuracy Comparison (From Epoch {acc_start_epoch}, acc=ave[-{window_size}:])', fontsize=16)
# plt.xlabel('Epoch', fontsize=16)
# plt.ylabel('Accuracy', fontsize=16)
# plt.legend(fontsize=18)  # 设置图例字体大小
# plt.grid(True)
# plt.tight_layout()
# if fix_params == 'T':
#     acc_plot_path = f'{save_dir}{file_name}_{model_name}_{dataset_name}_fix_acc.png'
# else:
#     acc_plot_path = f'{save_dir}{file_name}_{model_name}_{dataset_name}_acc.png'
# plt.savefig(acc_plot_path, dpi=300, bbox_inches='tight')
# plt.close()
# print(f"准确率对比图已保存到: {acc_plot_path}")

# ################################   AdaQHN 不同算法在特殊参数选择下对比     ###################################
"""
AdaQHN

各算法超参数分别为Adam[0.9, 0.999]; Adam_win[0.9, 0.999]; Adan[0.96, 0.9, 0.99]; 
QHAdam[0.999, 0.999, 0.7]; AdaQHN[0.999, 0.999, 0.7]

在['PoolMLP', 'EMnist_digits'], lr分别为[0.02, 0.02, 0.03, 0.02, 0.015]
在['GLeNet5', 'FMnist'], lr分别为[0.007, 0.01, 0.007, 0.01, 0.01]
在['TinyVGG', 'EMnist_letters'], lr分别为[0.01, 0.015, 0.015, 0.02, 0.015]
在['MicroVGG', 'cifar10'], lr分别为[0.02, 0.03, 0.01, 0.05, 0.03]
在['GRU1', 'AGnews'], lr分别为[0.007, 0.007, 0.007, 0.005, 0.001]
在['LSTM1', 'AGnews'], lr分别为[0.0007, 0.0007, 0.003, 0.005, 0.005]
"""

# model_name, dataset_name, optimizer= 'LSTM1', 'AGnews', 'AdaQHN'  # 模型, 数据集, 算法
# lr_list = [0.0007, 0.0007, 0.003, 0.005, 0.005]
# window_size = 10  # 使用最后window_size个数据点计算均值
# loss_start_epoch = 10  # 从第 loss_start_epoch 个数据点开始绘制loss图
# decimal_places = 5  # 设置保留的小数位数，可以根据需要调整
#
# plot_data = {'values': [], 'labels': []}
#
# Adam_file = f'../save_data/{optimizer}_{model_name}_{dataset_name}/ave_Adam_(0.9, 0.999)_{lr_list[0]}.pkl'
# with open(Adam_file, 'rb') as f:
#     Adam_data = pickle.load(f)
# Adam_train_loss_list = np.array(Adam_data['train_loss_list'])
# Adam_min_loss = round(np.mean(Adam_train_loss_list[-window_size:]), decimal_places)
# plot_data['values'].append(Adam_train_loss_list[loss_start_epoch:])
# plot_data['labels'].append(f'Adam(0.9, 0.999), loss={Adam_min_loss}')
#
# Adam_win_file = f'../save_data/{optimizer}_{model_name}_{dataset_name}/ave_Adam_win_(0.9, 0.999)_{lr_list[1]}.pkl'
# with open(Adam_win_file, 'rb') as f:
#     Adam_win_data = pickle.load(f)
# Adam_win_train_loss_list = np.array(Adam_win_data['train_loss_list'])
# Adam_win_min_loss = round(np.mean(Adam_win_train_loss_list[-window_size:]), decimal_places)
# plot_data['values'].append(Adam_win_train_loss_list[loss_start_epoch:])
# plot_data['labels'].append(f'Adam_win(0.9, 0.999), loss={Adam_win_min_loss}')
#
# Adan_file = f'../save_data/{optimizer}_{model_name}_{dataset_name}/ave_Adan_(0.96, 0.9, 0.99)_{lr_list[2]}.pkl'
# with open(Adan_file, 'rb') as f:
#     Adan_data = pickle.load(f)
# Adan_train_loss_list = np.array(Adan_data['train_loss_list'])
# Adan_min_loss = round(np.mean(Adan_train_loss_list[-window_size:]), decimal_places)
# plot_data['values'].append(Adan_train_loss_list[loss_start_epoch:])
# plot_data['labels'].append(f'Adan(0.96, 0.9, 0.99), loss={Adan_min_loss}')
#
# QHAdam_file = f'../save_data/{optimizer}_{model_name}_{dataset_name}/ave_QHAdam_(0.999, 0.999, 0.7)_{lr_list[3]}.pkl'
# with open(QHAdam_file, 'rb') as f:
#     QHAdam_data = pickle.load(f)
# QHAdam_train_loss_list = np.array(QHAdam_data['train_loss_list'])
# QHAdam_min_loss = round(np.mean(QHAdam_train_loss_list[-window_size:]), decimal_places)
# plot_data['values'].append(QHAdam_train_loss_list[loss_start_epoch:])
# plot_data['labels'].append(f'QHAdam(0.999, 0.999, 0.7), loss={QHAdam_min_loss}')
#
# AdaQHN_file = f'../save_data/{optimizer}_{model_name}_{dataset_name}/ave_AdaQHN_(0.999, 0.999, 0.7)_{lr_list[4]}.pkl'
# with open(AdaQHN_file, 'rb') as f:
#     AdaQHN_data = pickle.load(f)
# AdaQHN_train_loss_list = np.array(AdaQHN_data['train_loss_list'])
# AdaQHN_min_loss = round(np.mean(AdaQHN_train_loss_list[-window_size:]), decimal_places)
# plot_data['values'].append(AdaQHN_train_loss_list[loss_start_epoch:])
# plot_data['labels'].append(f'AdaQHN(0.999, 0.999, 0.7), loss={AdaQHN_min_loss}')
#
#
# # 绘制损失对比图
# plt.figure(figsize=(8, 6))
# for i, (values, label) in enumerate(zip(plot_data['values'], plot_data['labels'])):
#     plt.plot(range(loss_start_epoch, loss_start_epoch + len(values)), values, label=label, linewidth=2)
# plt.title(f'Training Loss Comparison (From Epoch {loss_start_epoch}, loss=ave[-{window_size}:])', fontsize=16)
# plt.xlabel('Epoch', fontsize=16)
# plt.ylabel('Training Loss', fontsize=16)
# plt.legend(fontsize=16)  # 设置图例字体大小
# plt.grid(True)
# plt.tight_layout()
# plot_path = f'../save_tu/{optimizer}_{model_name}_{dataset_name}_fix_loss.png'
# plt.savefig(plot_path, dpi=300, bbox_inches='tight')
# plt.close()
# print(f"图已保存到: {plot_path}")


# ################################   QHN和AdaQHN的参数敏感性     ###################################
"""
QHM和QHN[beta[0.95, 0.99, 0.995, 0.999, 0.9995], v[0.5, 0.6, 0.7, 0.8, 0.9]]
在['TinyVGG', 'EMnist_letters']上，第二次lr搜索范围为[0.7, 1, 1.2, 1.5, 2]

AdaQHN[beta[0.95, 0.99, 0.995, 0.999, 0.9995], 0.999, v[0.5, 0.6, 0.7, 0.8, 0.9]]
在['TinyVGG', 'EMnist_letters']上，第二次lr搜索范围为[0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05]
"""

model_name, dataset_name, file_name= 'TinyVGG', 'EMnist_letters', 'QHN'  # 模型，数据集和算法类
optimizer_list = ['QHM', 'QHN']
fix_param_name = 'beta'
fix_param_value = 0.9995
# optimizer_list = ['QHM', 'QHN', 'AdaQHN']

window_size = 10  # 使用最后window_size个数据点计算均值
loss_start_epoch = 10  # 从第 loss_start_epoch 个数据点开始绘制loss图
acc_start_epoch = 10  # 从第 acc_start_epoch 个数据点开始绘制acc图
decimal_places = 5  # 设置保留的小数位数，可以根据需要调整

# 用于存储绘图数据的字典
plot_data = {'values': [], 'labels': []}

for optimizer in optimizer_list:

    if optimizer == 'QHM':
        # 原始参数范围
        beta_list = [0.95, 0.995, 0.9995]
        v_list = [0.5, 0.7, 0.9]
        lr_list = [0.7, 1, 1.2, 1.5, 2]
        if fix_param_name == 'beta':
            beta_list = [fix_param_value]
        elif fix_param_name == 'v':
            v_list = [fix_param_value]
        else:
            raise ValueError(f"Unknown fix_param_name: {fix_param_name}")
        parameters_list = [beta_list, v_list]

    elif optimizer == 'QHN':
        beta_list = [0.95, 0.995, 0.9995]
        v_list = [0.5, 0.7, 0.9]
        lr_list = [0.7, 1, 1.2, 1.5, 2]
        if fix_param_name == 'beta':
            beta_list = [fix_param_value]
        elif fix_param_name == 'v':
            v_list = [fix_param_value]
        else:
            raise ValueError(f"Unknown fix_param_name: {fix_param_name}")
        parameters_list = [beta_list, v_list]

    elif optimizer == 'AdaQHN':
        beta1_list = [0.95, 0.99, 0.995, 0.999, 0.9995]
        beta2_list = [0.999]
        v_list = [0.5, 0.6, 0.7, 0.8, 0.9]
        lr_list = [0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05]
        if fix_param_name == 'beta1':
            beta1_list = [fix_param_value]
        elif fix_param_name == 'beta2':
            beta2_list = [fix_param_value]
        elif fix_param_name == 'v':
            v_list = [fix_param_value]
        else:
            raise ValueError(f"Unknown fix_param_name: {fix_param_name}")
        parameters_list = [beta1_list, beta2_list, v_list]

    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    for parameters in itertools.product(*parameters_list):

        min_loss = 1e4
        for lr in lr_list:
            file = f'../save_data/{file_name}_{model_name}_{dataset_name}/ave_{optimizer}_{parameters}_{lr}.pkl'

            try:
                with open(file, 'rb') as f:
                    data = pickle.load(f)
                train_loss_list = np.array(data['train_loss_list'])
                min_lin_loss = np.mean(train_loss_list[-window_size:])
                # 更新最小损失
                if min_lin_loss < min_loss:
                    min_loss = min_lin_loss
                    loss_params_file = file
                    best_loss_data = data  # 保存最小loss对应的数据
            except FileNotFoundError:
                print(f"File not found: {file}")
                continue

        formatted_min_loss = round(min_loss, decimal_places)  # 保留至小数点后5位
        label = f"{optimizer} {parameters}, loss={formatted_min_loss}"

        plot_data['values'].append(best_loss_data['train_loss_list'][loss_start_epoch:])
        plot_data['labels'].append(label)


# 绘制损失对比图
plt.figure(figsize=(8, 6))
for i, (values, label) in enumerate(zip(plot_data['values'], plot_data['labels'])):
    plt.plot(range(loss_start_epoch, loss_start_epoch + len(values)), values, label=label, linewidth=2)
plt.title(f'Parameter ablation', fontsize=16)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Training Loss', fontsize=16)
plt.legend(fontsize=16)  # 设置图例字体大小
plt.grid(True)
plt.tight_layout()
plot_path = f'../save_tu/ablation_{model_name}_{dataset_name}_{file_name}_{fix_param_name}={fix_param_value}.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"图已保存到: {plot_path}")


# ##########################  局部放大图代码  ########################################################

# file1 = '../save_data/QHN_MLP1_DXS1/ave_SGD_beta0.9_v1_lr1.pkl'
# with open(file1, 'rb') as f:
#     data1 = pickle.load(f)
# yi_train = np.array(data1['train_loss_list'])
# yi_validation = np.array(data1['validation_loss_list'])
#
# file2 = '../save_data/QHN_MLP1_DXS1/ave_momentum_beta0.95_v1_lr1.pkl'
# with open(file2, 'rb') as f:
#     data2 = pickle.load(f)
# er_train = np.array(data2['train_loss_list'])
# er_validation = np.array(data2['validation_loss_list'])
#
# file3 = '../save_data/QHN_MLP1_DXS1/ave_NAG_beta0.95_v1_lr1.pkl'
# with open(file3, 'rb') as f:
#     data3 = pickle.load(f)
# san_train = np.array(data3['train_loss_list'])
# san_validation = np.array(data3['validation_loss_list'])
#
# file4 = '../save_data/QHN_MLP1_DXS1/ave_RNAG_beta0.95_v1_lr1.pkl'
# with open(file4, 'rb') as f:
#     data4 = pickle.load(f)
# si_train = np.array(data4['train_loss_list'])
# si_validation = np.array(data4['validation_loss_list'])
#
# file5 = '../save_data/QHN_MLP1_DXS1/ave_QHM_beta0.99_v0.95_lr1.pkl'
# with open(file5, 'rb') as f:
#     data5 = pickle.load(f)
# wu_train = np.array(data5['train_loss_list'])
# wu_validation = np.array(data5['validation_loss_list'])
#
# file6 = '../save_data/QHN_MLP1_DXS1/ave_QHN_beta0.99_v0.95_lr1.pkl'
# with open(file6, 'rb') as f:
#     data6 = pickle.load(f)
# liu_train = np.array(data6['train_loss_list'])
# liu_validation = np.array(data6['validation_loss_list'])
#
#
# x = np.arange(1, 151)
# # 训练损失
# fig, ax = plt.subplots(1, 1)
# ax.plot(x, yi_train, label='SGD')
# ax.plot(x, er_train, label='Momentum best(β=0.95)')
# ax.plot(x, san_train, label='NAG best(β=0.95)')
# ax.plot(x, si_train, label='RNAG best(β=0.95)')
# ax.plot(x, wu_train, label='QHM best(β=0.99 v=0.95)')
# ax.plot(x, liu_train, label='QHN best(β=0.99 v=0.95)')
# # ax.set_title('lr=1, Train Loss ~ Train Epoch')  # 设置标题
# ax.set_xlabel('Epoch')  # 设置 x 轴名字
# ax.set_ylabel('Train Loss')	 # 设置 y 轴名字
# ax.legend(loc="upper right")  # 调整图例位置
# ax.grid(True)  # 添加网格线
# # 绘制子图
# axins = inset_axes(ax, width="40%", height="30%", loc='center',
#                    bbox_to_anchor=(0.3, -0.1, 1, 1),
#                    bbox_transform=ax.transAxes)
#
# axins.plot(x, yi_train)
# axins.plot(x, er_train)
# axins.plot(x, san_train)
# axins.plot(x, si_train)
# axins.plot(x, wu_train)
# axins.plot(x, liu_train)
# # axins.grid(True)  # 添加网格线
# # 设置放大区间
# zone_left = 100
# zone_right = 149
# # 坐标轴的扩展比例（根据实际数据调整）
# x_ratio = 0  # x轴显示范围的扩展比例
# y_ratio = 0.2  # y轴显示范围的扩展比例
# # X轴的显示范围
# xlim0 = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio
# xlim1 = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio
# # Y轴的显示范围
# y = np.hstack((
#                # yi_train[zone_left:zone_right],
#                er_train[zone_left:zone_right],
#                san_train[zone_left:zone_right],
#                si_train[zone_left:zone_right],
#                wu_train[zone_left:zone_right],
#                liu_train[zone_left:zone_right],
#                ))
# ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
# ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio
# # 调整子坐标系的显示范围
# axins.set_xlim(xlim0, xlim1)
# axins.set_ylim(ylim0, ylim1)
# # 原图中画方框
# tx0 = xlim0
# tx1 = xlim1
# ty0 = ylim0
# ty1 = ylim1
# sx = [tx0, tx1, tx1, tx0, tx0]
# sy = [ty0, ty0, ty1, ty1, ty0]
# ax.plot(sx, sy, "black")
# # 画两条线
# xy = (xlim0, ylim1)
# xy2 = (xlim0, ylim0)
# con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
#                       axesA=axins, axesB=ax)
# axins.add_artist(con)
# xy = (xlim1, ylim1)
# xy2 = (xlim1, ylim0)
# con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
#                       axesA=axins, axesB=ax)
# axins.add_artist(con)
# # 画图
# plt.show()
#
# # 验证精度
# fig, ax = plt.subplots(1, 1)
# ax.plot(x, yi_validation, label='SGD')
# ax.plot(x, er_validation, label='Momentum β=0.95')
# ax.plot(x, san_validation, label='NAG β=0.95')
# ax.plot(x, si_validation, label='RNAG β=0.95')
# ax.plot(x, wu_validation, label='QHM β=0.99 v=0.95')
# ax.plot(x, liu_validation, label='QHN β=0.99 v=0.95')
# # ax.set_title('lr=1, Train Loss ~ Train Epoch')  # 设置标题
# ax.set_xlabel('Epoch')  # 设置 x 轴名字
# ax.set_ylabel('Validation Loss')	 # 设置 y 轴名字
# ax.legend(loc="upper right")  # 调整图例位置  upper,lower/lift,right
# ax.grid(True)  # 添加网格线
# # 绘制子图
# axins = inset_axes(ax, width="60%", height="30%", loc='center',
#                    bbox_to_anchor=(0.2, -0.1, 1, 1),
#                    bbox_transform=ax.transAxes)
#
# axins.plot(x, yi_validation)
# axins.plot(x, er_validation)
# axins.plot(x, san_validation)
# axins.plot(x, si_validation)
# axins.plot(x, wu_validation)
# axins.plot(x, liu_validation)
# # axins.grid(True)  # 添加网格线
# # 设置放大区间
# zone_left = 50
# zone_right = 149
# # 坐标轴的扩展比例（根据实际数据调整）
# x_ratio = 0  # x轴显示范围的扩展比例
# y_ratio = 0.2  # y轴显示范围的扩展比例
# # X轴的显示范围
# xlim0 = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio
# xlim1 = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio
# # Y轴的显示范围
# y = np.hstack((
#                # yi_validation[zone_left:zone_right],
#                er_validation[zone_left:zone_right],
#                san_validation[zone_left:zone_right],
#                si_validation[zone_left:zone_right],
#                wu_validation[zone_left:zone_right],
#                liu_validation[zone_left:zone_right],
#                ))
# ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
# ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio
# # 调整子坐标系的显示范围
# axins.set_xlim(xlim0, xlim1)
# axins.set_ylim(ylim0, ylim1)
# # 原图中画方框
# tx0 = xlim0
# tx1 = xlim1
# ty0 = ylim0
# ty1 = ylim1
# sx = [tx0, tx1, tx1, tx0, tx0]
# sy = [ty0, ty0, ty1, ty1, ty0]
# ax.plot(sx, sy, "black")
# # 画两条线
# xy = (xlim0, ylim1)
# xy2 = (xlim0, ylim0)
# con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
#                       axesA=axins, axesB=ax)
# axins.add_artist(con)
# xy = (xlim1, ylim1)
# xy2 = (xlim1, ylim0)
# con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
#                       axesA=axins, axesB=ax)
# axins.add_artist(con)
# # 画图
# plt.show()

