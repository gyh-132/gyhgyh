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
QHM,QHN[[0.8, 0.85, 0.9, 0.95, 0.99], [0.5, 0.6, 0.7, 0.8, 0.9]]

在['PoolMLP', 'EMnist_digits']上，第二次lr搜索范围分别为SGD[0.5, 0.7, 1, 1.2, 1.5]，其它[2, 2.5, 3, 3.5, 4]

在['TinyVGG', 'EMnist_letters']上，第二次lr搜索范围分别为SGD[0.15, 0.2, 0.3, 0.5, 0.7]，其它[0.7, 1, 1.2, 1.5, 2]

在['MobileCNN', 'EMnist_balanced']上，第二次lr搜索范围分别为SGD[0.2, 0.3, 0.5, 0.7, 1];其它[0.7, 1, 1.2, 1.5, 2]
"""

# file_name = 'QHN'
# model_name, dataset_name, = 'PoolMLP', 'EMnist_digits'  # 模型、数据集
# # opt_list = ['SGD', 'momentum', 'NAG', 'QHM', 'QHN']
# opt_list = ['SGD', 'momentum', 'NAG', 'QHM', 'QHN']
#
# save_dir = f'../save_tu/{file_name}_{model_name}_{dataset_name}/'
# os.makedirs(save_dir, exist_ok=True)
# with open(save_dir + "select_params.txt", "w") as file:  # 先清空文件
#     pass  # 只是确保文件存在且为空
#
# window_size = 10  # 使用最后window_size个数据点计算均值
# loss_start_epoch = 20  # 从第 loss_start_epoch 个数据点开始绘制loss图
# acc_start_epoch = 20   # 从第 acc_start_epoch 个数据点开始绘制acc图
# loss_limit = ['1', 0.03]   # T
# acc_limit = ['1', 0.92]
# decimal_places = 5  # 设置保留的小数位数，可以根据需要调整
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
#         lr_list = [0.5, 0.7, 1, 1.2, 1.5]
#     elif optimizer == 'momentum':
#         beta_list = [0.8, 0.85, 0.9, 0.95, 0.99]
#         parameters_list = [beta_list]
#         lr_list = [2, 2.5, 3, 3.5, 4]
#     elif optimizer == 'NAG':
#         beta_list = [0.8, 0.85, 0.9, 0.95, 0.99]
#         parameters_list = [beta_list]
#         lr_list = [2, 2.5, 3, 3.5, 4]
#     elif optimizer == 'QHM':
#         beta_list = [0.8, 0.85, 0.9, 0.95, 0.99]
#         v_list = [0.5, 0.6, 0.7, 0.8, 0.9]
#         parameters_list = [beta_list, v_list]
#         lr_list = [2, 2.5, 3, 3.5, 4]
#     elif optimizer == 'QHN':
#         beta_list = [0.8, 0.85, 0.9, 0.95, 0.99]
#         v_list = [0.5, 0.6, 0.7, 0.8, 0.9]
#         parameters_list = [beta_list, v_list]
#         lr_list = [2, 2.5, 3, 3.5, 4]
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
#         with open(save_dir + "select_params.txt", "a") as file:  # 注意改为 "a" 模式
#             file.write(f"{optimizer} loss_file: {loss_params_file} loss: {formatted_min_loss}\n")
#             file.write(f"{optimizer} acc_file: {acc_params_file} acc: {formatted_max_acc}\n")
#
#         # 存储绘图数据，并在标签中添加min_loss和max_acc
#         loss_label = f"{optimizer} (min_loss={formatted_min_loss})"
#         acc_label = f"{optimizer} (max_acc={formatted_max_acc})"
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
# plt.title(f'Training Loss Comparison (From Epoch {loss_start_epoch}, min_loss=ave[-{window_size}:])')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# loss_plot_path = f'{save_dir}optimizers_loss_comparison.png'
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
# plt.title(f'Validation Accuracy Comparison (From Epoch {acc_start_epoch}, max_acc=ave[-{window_size}:])')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# acc_plot_path = f'{save_dir}optimizers_acc_comparison.png'
# plt.savefig(acc_plot_path, dpi=300, bbox_inches='tight')
# plt.close()
# print(f"准确率对比图已保存到: {acc_plot_path}")

# ##########################  QHN与QHM的部分参数对此图  ########################################################
"""
QHN

QHM与QHN超参数搜索范围为[[0.8, 0.85, 0.9, 0.95, 0.99], [0.5, 0.6, 0.7, 0.8, 0.9]]

在['PoolMLP', 'EMnist_digits']上，第二次lr搜索范围均为[2, 2.5, 3, 3.5, 4]

在['TinyVGG', 'EMnist_letters']上，第二次lr搜索范围均为[0.7, 1, 1.2, 1.5, 2]

在['MobileCNN', 'EMnist_balanced']上，第二次lr搜索范围均为[0.7, 1, 1.2, 1.5, 2]
"""

# file_name = 'QHN'
# model_name, dataset_name, = 'MobileCNN', 'EMnist_balanced'  # 模型、数据集
# opt_list = ['QHM', 'QHN']
# params = [[0.9, 0.5], [0.9, 0.9]]
# lr = 1
# start_epoch = 70  # 从第 start_epoch 个数据点开始绘制loss图
#
# data_path_1 = f'../save_data/{file_name}_{model_name}_{dataset_name}/ave_QHM_({params[0][0]}, {params[0][1]})_{lr}.pkl'
# # data_path_1 = f'../save_data/{file_name}_{model_name}_{dataset_name}/ave_QHM_(0.9, 0.9)_1.pkl'
# with open(data_path_1, 'rb') as f:
#     data = pickle.load(f)
# loss_list_1 = np.array(data['train_loss_list'][start_epoch:])
#
# data_path_2 = f'../save_data/{file_name}_{model_name}_{dataset_name}/ave_QHM_({params[1][0]}, {params[1][1]})_{lr}.pkl'
# with open(data_path_2, 'rb') as f:
#     data = pickle.load(f)
# loss_list_2 = np.array(data['train_loss_list'][start_epoch:])
#
# data_path_3 = f'../save_data/{file_name}_{model_name}_{dataset_name}/ave_QHN_({params[0][0]}, {params[0][1]})_{lr}.pkl'
# with open(data_path_3, 'rb') as f:
#     data = pickle.load(f)
# loss_list_3 = np.array(data['train_loss_list'][start_epoch:])
#
# data_path_4 = f'../save_data/{file_name}_{model_name}_{dataset_name}/ave_QHN_({params[1][0]}, {params[1][1]})_{lr}.pkl'
# with open(data_path_4, 'rb') as f:
#     data = pickle.load(f)
# loss_list_4 = np.array(data['train_loss_list'][start_epoch:])
#
# plt.figure(figsize=(8, 6))
# plt.plot(range(start_epoch, start_epoch + len(loss_list_1)), loss_list_1,
#          label=f'QHM, β={params[0][0]}, v={params[0][1]}, lr={lr}', linewidth=1.5)
# plt.plot(range(start_epoch, start_epoch + len(loss_list_3)), loss_list_3,
#          label=f'QHN, β={params[0][0]}, v={params[0][1]}, lr={lr}', linewidth=1.5)
# plt.plot(range(start_epoch, start_epoch + len(loss_list_2)), loss_list_2,
#          label=f'QHM, β={params[1][0]}, v={params[1][1]}, lr={lr}', linewidth=1.5)
# plt.plot(range(start_epoch, start_epoch + len(loss_list_4)), loss_list_4,
#          label=f'QHN, β={params[1][0]}, v={params[1][1]}, lr={lr}', linewidth=1.5)
# plt.title(f'Train Loss Comparison (From Epoch {start_epoch})')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.show()

# ################################   AdaQHN 为各算法选择最优参数，按最低loss和最高acc     ###################################
"""
AdaQHN

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
Adam、Adam_win[0.0001, 0.00015, 0.0002, 0.0003, 0.0005, 0.0007, 0.001];
Adan[0.0005, 0.0007, 0.001, 0.0015, 0.002, 0.003, 0.005];
QHMAdam和AdaQHN[0.001, 0.0015, 0.002, 0.003, 0.005, 0.007, 0.01]
"""

# file_name = 'AdaQHN'
# model_name, dataset_name, = 'LSTM1', 'AGnews'  # 模型、数据集
# opt_list = ['Adam', 'Adam_win', 'Adan', 'QHAdam', 'AdaQHN']
#
# save_dir = f'../save_tu/{file_name}_{model_name}_{dataset_name}/'
# os.makedirs(save_dir, exist_ok=True)
# with open(save_dir + "select_params.txt", "w") as file:  # 先清空文件
#     pass  # 只是确保文件存在且为空
#
# window_size = 10  # 使用最后window_size个数据点计算均值
# loss_start_epoch = 10  # 从第 loss_start_epoch 个数据点开始绘制loss图
# acc_start_epoch = 10  # 从第 acc_start_epoch 个数据点开始绘制acc图
# loss_limit = ['1', 0.03]  # T
# acc_limit = ['1', 0.92]
# decimal_places = 5  # 设置保留的小数位数，可以根据需要调整
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
#         beta1_list = [0.8, 0.85, 0.9, 0.95, 0.99]
#         beta2_list = [0.999]
#         parameters_list = [beta1_list, beta2_list]
#         lr_list = [0.0001, 0.00015, 0.0002, 0.0003, 0.0005, 0.0007, 0.001]
#     elif optimizer == 'Adam_win':
#         beta1_list = [0.8, 0.85, 0.9, 0.95, 0.99]
#         beta2_list = [0.999]
#         parameters_list = [beta1_list, beta2_list]
#         lr_list = [0.0001, 0.00015, 0.0002, 0.0003, 0.0005, 0.0007, 0.001]
#     elif optimizer == 'Adan':
#         beta1_list = [0.9, 0.93, 0.96, 0.99]
#         beta2_list = [0.9, 0.93, 0.96, 0.99]
#         beta3_list = [0.9, 0.93, 0.96, 0.99]
#         parameters_list = [beta1_list, beta2_list, beta3_list]
#         lr_list = [0.0005, 0.0007, 0.001, 0.0015, 0.002, 0.003, 0.005]
#     elif optimizer == 'QHAdam':
#         beta1_list = [0.95, 0.99, 0.995, 0.999, 0.9995]
#         beta2_list = [0.999]
#         v_list = [0.5, 0.6, 0.7, 0.8, 0.9]
#         parameters_list = [beta1_list, beta2_list, v_list]
#         lr_list = [0.001, 0.0015, 0.002, 0.003, 0.005, 0.007, 0.01]
#     elif optimizer == 'AdaQHN':
#         beta1_list = [0.95, 0.99, 0.995, 0.999, 0.9995]
#         beta2_list = [0.999]
#         v_list = [0.5, 0.6, 0.7, 0.8, 0.9]
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
#         with open(save_dir + "select_params.txt", "a") as file:  # 注意改为 "a" 模式
#             file.write(f"{optimizer} loss_file: {loss_params_file} loss: {formatted_min_loss}\n")
#             file.write(f"{optimizer} acc_file: {acc_params_file} acc: {formatted_max_acc}\n")
#
#         # 存储绘图数据，并在标签中添加min_loss和max_acc
#         loss_label = f"{optimizer} (min_loss={formatted_min_loss})"
#         acc_label = f"{optimizer} (max_acc={formatted_max_acc})"
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
# plt.title(f'Training Loss Comparison (From Epoch {loss_start_epoch}, min_loss=ave[-{window_size}:])')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# loss_plot_path = f'{save_dir}optimizers_loss_comparison.png'
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
# plt.title(f'Validation Accuracy Comparison (From Epoch {acc_start_epoch}, max_acc=ave[-{window_size}:])')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# acc_plot_path = f'{save_dir}optimizers_acc_comparison.png'
# plt.savefig(acc_plot_path, dpi=300, bbox_inches='tight')
# plt.close()
# print(f"准确率对比图已保存到: {acc_plot_path}")


# ##########################  局部放大图  ########################################################

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

