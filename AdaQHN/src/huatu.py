import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.patches import ConnectionPatch

##########################  QHN  ########################################################

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
# ax.plot(x, wu_train, label='QHM best(β=0.99 v=0.95)')
# ax.plot(x, si_train, label='RNAG best(β=0.95)')
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
# axins.plot(x, wu_train)
# axins.plot(x, si_train)
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
# ax.plot(x, wu_validation, label='QHM β=0.99 v=0.95')
# ax.plot(x, si_validation, label='RNAG β=0.95')
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
# axins.plot(x, wu_validation)
# axins.plot(x, si_validation)
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

############################# AdaQHN #####################################################

# file1 = '../save_data/AdaQHN_GRU1_PTB/ave_Adam_lr0.003.pkl'
# with open(file1, 'rb') as f:
#     data1 = pickle.load(f)
# yi_train = np.array(data1['train_loss_list'])
# yi_validation = np.array(data1['validation_loss_list'])
#
# file2 = '../save_data/AdaQHN_GRU1_PTB/ave_Adam_win_lr0.003.pkl'
# with open(file2, 'rb') as f:
#     data2 = pickle.load(f)
# er_train = np.array(data2['train_loss_list'])
# er_validation = np.array(data2['validation_loss_list'])
#
# file3 = '../save_data/AdaQHN_GRU1_PTB/ave_Adan_lr0.01.pkl'
# with open(file3, 'rb') as f:
#     data3 = pickle.load(f)
# san_train = np.array(data3['train_loss_list'])
# san_validation = np.array(data3['validation_loss_list'])
#
# file4 = '../save_data/AdaQHN_GRU1_PTB/ave_QHAdam_lr0.007.pkl'
# with open(file4, 'rb') as f:
#     data4 = pickle.load(f)
# si_train = np.array(data4['train_loss_list'])
# si_validation = np.array(data4['validation_loss_list'])
#
# file5 = '../save_data/AdaQHN_GRU1_PTB/ave_AdaQHN_lr0.007.pkl'
# with open(file5, 'rb') as f:
#     data5 = pickle.load(f)
# wu_train = np.array(data5['train_loss_list'])
# wu_validation = np.array(data5['validation_loss_list'])
#
#
# x = np.arange(1, 81)
# # 训练损失
# fig, ax = plt.subplots(1, 1)
# ax.plot(x, yi_train, label='Adam best(lr=0.003)')
# ax.plot(x, er_train, label='Adam_win best(lr=0.003)')
# ax.plot(x, san_train, label='Adan best(lr=0.01)')
# ax.plot(x, si_train, label='QHAdam best(lr=0.007)')
# ax.plot(x, wu_train, label='AdaQHN best(lr=0.007)')
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
# # axins.grid(True)  # 添加网格线
# # 设置放大区间
# zone_left = 69
# zone_right = 79
# # 坐标轴的扩展比例（根据实际数据调整）
# x_ratio = 0  # x轴显示范围的扩展比例
# y_ratio = 0.2  # y轴显示范围的扩展比例
# # X轴的显示范围
# xlim0 = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio
# xlim1 = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio
# # Y轴的显示范围
# y = np.hstack((
#                yi_train[zone_left:zone_right],
#                er_train[zone_left:zone_right],
#                san_train[zone_left:zone_right],
#                si_train[zone_left:zone_right],
#                wu_train[zone_left:zone_right],
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
# ax.plot(x, yi_validation, label='Adam lr=0.003')
# ax.plot(x, er_validation, label='Adam_win lr=0.003')
# ax.plot(x, san_validation, label='Adan lr=0.01')
# ax.plot(x, si_validation, label='QHAdam lr=0.007')
# ax.plot(x, wu_validation, label='AdaQHN lr=0.007')
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
# # axins.grid(True)  # 添加网格线
# # 设置放大区间
# zone_left = 30
# zone_right = 79
# # 坐标轴的扩展比例（根据实际数据调整）
# x_ratio = 0  # x轴显示范围的扩展比例
# y_ratio = 0.2  # y轴显示范围的扩展比例
# # X轴的显示范围
# xlim0 = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio
# xlim1 = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio
# # Y轴的显示范围
# y = np.hstack((
#                yi_validation[zone_left:zone_right],
#                er_validation[zone_left:zone_right],
#                san_validation[zone_left:zone_right],
#                si_validation[zone_left:zone_right],
#                wu_validation[zone_left:zone_right],
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

############################# Ada_algorithm_QHN #####################################################

# file1 = '../save_data/Ada_algorithm_QHN_GRU1_PTB/ave_Adabelief_lr0.003.pkl'
# with open(file1, 'rb') as f:
#     data1 = pickle.load(f)
# yi_train = np.array(data1['train_loss_list'])
# yi_validation = np.array(data1['validation_loss_list'])
#
# file2 = '../save_data/Ada_algorithm_QHN_GRU1_PTB/ave_Adabelief_QHN_lr0.05.pkl'
# with open(file2, 'rb') as f:
#     data2 = pickle.load(f)
# er_train = np.array(data2['train_loss_list'])
# er_validation = np.array(data2['validation_loss_list'])
#
# file3 = '../save_data/Ada_algorithm_QHN_GRU1_PTB/ave_YOGI_lr0.01.pkl'
# with open(file3, 'rb') as f:
#     data3 = pickle.load(f)
# san_train = np.array(data3['train_loss_list'])
# san_validation = np.array(data3['validation_loss_list'])
#
# file4 = '../save_data/Ada_algorithm_QHN_GRU1_PTB/ave_YOGI_QHN_lr0.01.pkl'
# with open(file4, 'rb') as f:
#     data4 = pickle.load(f)
# si_train = np.array(data4['train_loss_list'])
# si_validation = np.array(data4['validation_loss_list'])
#
#
# x = np.arange(1, 81)
# # 训练损失
# fig, ax = plt.subplots(1, 1)
# ax.plot(x, yi_train, label='Adabelief best(lr=0.003)')
# ax.plot(x, er_train, label='Adabelief_QHN best(lr=0.05)')
# ax.plot(x, san_train, label='YOGI best(lr=0.01)')
# ax.plot(x, si_train, label='YOGI_QHN best(lr=0.01)')
# # ax.set_title('lr=1, Train Loss ~ Train Epoch')  # 设置标题
# ax.set_xlabel('Epoch')  # 设置 x 轴名字
# ax.set_ylabel('Train Loss')	 # 设置 y 轴名字
# ax.legend(loc="upper right")  # 调整图例位置
# ax.grid(True)  # 添加网格线
# # 绘制子图
# axins = inset_axes(ax, width="40%", height="30%", loc='center',
#                    bbox_to_anchor=(0.3, 0, 1, 1),
#                    bbox_transform=ax.transAxes)
#
# axins.plot(x, yi_train)
# axins.plot(x, er_train)
# axins.plot(x, san_train)
# axins.plot(x, si_train)
# # axins.grid(True)  # 添加网格线
# # 设置放大区间
# zone_left = 69
# zone_right = 79
# # 坐标轴的扩展比例（根据实际数据调整）
# x_ratio = 0  # x轴显示范围的扩展比例
# y_ratio = 0.2  # y轴显示范围的扩展比例
# # X轴的显示范围
# xlim0 = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio
# xlim1 = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio
# # Y轴的显示范围
# y = np.hstack((
#                yi_train[zone_left:zone_right],
#                er_train[zone_left:zone_right],
#                san_train[zone_left:zone_right],
#                si_train[zone_left:zone_right],
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
# ax.plot(x, yi_validation, label='Adabelief lr=0.003')
# ax.plot(x, er_validation, label='Adabelief_QHN lr=0.05')
# ax.plot(x, san_validation, label='YOGI lr=0.01')
# ax.plot(x, si_validation, label='YOGI_QHN lr=0.01')
# # ax.set_title('lr=1, Train Loss ~ Train Epoch')  # 设置标题
# ax.set_xlabel('Epoch')  # 设置 x 轴名字
# ax.set_ylabel('Validation Loss')	 # 设置 y 轴名字  Validation Loss or Accuracy
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
# # axins.grid(True)  # 添加网格线
# # 设置放大区间
# zone_left = 30
# zone_right = 79
# # 坐标轴的扩展比例（根据实际数据调整）
# x_ratio = 0  # x轴显示范围的扩展比例
# y_ratio = 0.2  # y轴显示范围的扩展比例
# # X轴的显示范围
# xlim0 = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio
# xlim1 = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio
# # Y轴的显示范围
# y = np.hstack((
#                yi_validation[zone_left:zone_right],
#                er_validation[zone_left:zone_right],
#                san_validation[zone_left:zone_right],
#                si_validation[zone_left:zone_right],
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


#########################################################################################
# """
# 这部分代码用来重绘一些图片
# 比如Adam_win在  GLeNet FMnist lr=0.01时发生梯度爆炸，绘图时需要删除
# """
# # 绘图
# # opt_list = ['Adabelief', 'Adabelief_QHN', 'YOGI', 'YOGI_QHN']
# # opt_list = ['Adam', 'QHAdam', 'Adan', 'Adam_win', 'AdaQHN']
# # lr_list = [0.001, 0.0015, 0.002, 0.003, 0.005, 0.007, 0.01, 0.012, 0.015, 0.02]
# # lr_list = [0.001, 0.0015, 0.002, 0.003, 0.005, 0.007, 0.01]
# # file_name = 'AdaQHN' or 'Ada_algorithm_QHN'
# # model_name = 'MLP1' or 'GLeNet'
# # dataset_name = 'DXS1' or 'FMnist'
# file_name, model_name, dataset_name, optimizer = 'Ada_algorithm_QHN', 'GLeNet', 'FMnist', 'Adabelief'
# lr_list = [0.001, 0.0015, 0.002, 0.003, 0.005, 0.007]
# Epoch = 80
#
# colors = matplotlib.colormaps.get_cmap('tab10')  # 颜色映射
# plt.figure(figsize=(10, 6))
# for index in range(len(lr_list)):
#     file = '../save_data/{}_{}_{}/ave_{}_lr{}.pkl' \
#         .format(file_name, model_name, dataset_name, optimizer, lr_list[index])
#     with open(file, 'rb') as f:
#         data = pickle.load(f)
#     train_loss = np.array(data['train_loss_list'])
#     train_acc = np.array(data['train_acc_list'])
#     validation_loss = np.array(data['validation_loss_list'])
#     validation_acc = np.array(data['validation_acc_list'])
#
#     plt.plot(train_loss, label='{} lr{} train'
#              .format(optimizer, lr_list[index]), color=colors(index))
#     plt.plot(validation_loss, label='{} lr{} validation'
#              .format(optimizer, lr_list[index]), linestyle='--', color=colors(index))
# plt.plot(np.full(Epoch, 0), label='0')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid()
# plt.savefig('../save_tu/{}_{}_{}/{}_loss.png'.format(file_name, model_name, dataset_name, optimizer))
# plt.clf()  # 清除图像内容
# plt.close()  # 关闭图像
#
# plt.figure(figsize=(10, 6))
# for index in range(len(lr_list)):
#     file = '../save_data/{}_{}_{}/ave_{}_lr{}.pkl' \
#         .format(file_name, model_name, dataset_name, optimizer, lr_list[index])
#     with open(file, 'rb') as f:
#         data = pickle.load(f)
#     train_loss = np.array(data['train_loss_list'])
#     train_acc = np.array(data['train_acc_list'])
#     validation_loss = np.array(data['validation_loss_list'])
#     validation_acc = np.array(data['validation_acc_list'])
#
#     plt.plot(train_acc, label='{} lr{} train'
#              .format(optimizer, lr_list[index]), color=colors(index))
#     plt.plot(validation_acc, label='{} lr{} validation'
#              .format(optimizer, lr_list[index]), linestyle='--', color=colors(index))
# plt.plot(np.full(Epoch, 1), label='1')
# plt.xlabel('Epoch')
# plt.ylabel('ACC')
# plt.legend()
# plt.grid()
# plt.savefig('../save_tu/{}_{}_{}/{}_acc.png'.format(file_name, model_name, dataset_name, optimizer))
# plt.clf()  # 清除图像内容
# plt.close()  # 关闭图像

#########################################################################################
# v_list = [0.5, 0.7, 0.9, 0.99]
# colors = matplotlib.colormaps.get_cmap('tab10')  # 颜色映射
# plt.figure(figsize=(10, 6))
# for index in range(len(v_list)):
#     file1 = '../save_data/QHN_GLeNet_FMnist/ave_QHM_beta0.999_v{}_lr1.pkl'.format(v_list[index])
#     with open(file1, 'rb') as f:
#         data1 = pickle.load(f)
#     yi_train_loss = np.array(data1['train_loss_list'])
#
#     file2 = '../save_data/QHN_GLeNet_FMnist/ave_QHN_beta0.999_v{}_lr1.pkl'.format(v_list[index])
#     with open(file2, 'rb') as f:
#         data2 = pickle.load(f)
#     er_train_loss = np.array(data2['train_loss_list'])
#
#     plt.plot(yi_train_loss, label='QHM β=0.999 v={}'.format(v_list[index]), linestyle='--', color=colors(index))
#     plt.plot(er_train_loss, label='QHN β=0.999 v={}'.format(v_list[index]), color=colors(index))
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid()
# plt.show()

############################ QHN lr or v ########################################################################

file1 = '../save_data/QHN_MLP1_DXS1/ave_QHM_beta0.99_v0.95_lr0.5.pkl'
with open(file1, 'rb') as f:
    data1 = pickle.load(f)
yi_train_loss = np.array(data1['train_loss_list'])
file11 = '../save_data/QHN_MLP1_DXS1/ave_QHN_beta0.99_v0.95_lr0.5.pkl'
with open(file11, 'rb') as f:
    data11 = pickle.load(f)
yi1_train_loss = np.array(data11['train_loss_list'])

file2 = '../save_data/QHN_MLP1_DXS1/ave_QHM_beta0.99_v0.95_lr0.7.pkl'
with open(file2, 'rb') as f:
    data2 = pickle.load(f)
er_train_loss = np.array(data2['train_loss_list'])
file21 = '../save_data/QHN_MLP1_DXS1/ave_QHN_beta0.99_v0.95_lr0.7.pkl'
with open(file21, 'rb') as f:
    data21 = pickle.load(f)
er1_train_loss = np.array(data21['train_loss_list'])

file3 = '../save_data/QHN_MLP1_DXS1/ave_QHM_beta0.99_v0.95_lr1.pkl'
with open(file3, 'rb') as f:
    data3 = pickle.load(f)
san_train_loss = np.array(data3['train_loss_list'])
file31 = '../save_data/QHN_MLP1_DXS1/ave_QHN_beta0.99_v0.95_lr1.pkl'
with open(file31, 'rb') as f:
    data31 = pickle.load(f)
san1_train_loss = np.array(data31['train_loss_list'])

file4 = '../save_data/QHN_MLP1_DXS1/ave_QHM_beta0.99_v0.95_lr1.2.pkl'
with open(file4, 'rb') as f:
    data4 = pickle.load(f)
si_train_loss = np.array(data4['train_loss_list'])
file41 = '../save_data/QHN_MLP1_DXS1/ave_QHN_beta0.99_v0.95_lr1.2.pkl'
with open(file41, 'rb') as f:
    data41 = pickle.load(f)
si1_train_loss = np.array(data41['train_loss_list'])

file5 = '../save_data/QHN_MLP1_DXS1/ave_QHM_beta0.99_v0.95_lr1.5.pkl'
with open(file5, 'rb') as f:
    data5 = pickle.load(f)
wu_train_loss = np.array(data5['train_loss_list'])
file51 = '../save_data/QHN_MLP1_DXS1/ave_QHN_beta0.99_v0.95_lr1.5.pkl'
with open(file51, 'rb') as f:
    data51 = pickle.load(f)
wu1_train_loss = np.array(data51['train_loss_list'])

# file6 = '../save_data/QHN_MLP1_DXS1/ave_QHM_beta0.999_v0.99_lr1.pkl'
# with open(file6, 'rb') as f:
#     data6 = pickle.load(f)
# liu_train_loss = np.array(data6['train_loss_list'])
# file61 = '../save_data/QHN_MLP1_DXS1/ave_QHN_beta0.999_v0.99_lr1.pkl'
# with open(file61, 'rb') as f:
#     data61 = pickle.load(f)
# liu1_train_loss = np.array(data61['train_loss_list'])

colors = matplotlib.colormaps.get_cmap('tab10')  # 颜色映射
x = np.arange(1, 151)
# 训练损失
fig, ax = plt.subplots(1, 1)
ax.plot(x, yi_train_loss, label='QHM lr=0.5', linestyle='--', color=colors(0))
ax.plot(x, yi1_train_loss, label='QHN lr=0.5', color=colors(0))

ax.plot(x, er_train_loss, label='QHM lr=0.7', linestyle='--', color=colors(1))
ax.plot(x, er1_train_loss, label='QHM lr=0.7', color=colors(1))

ax.plot(x, san_train_loss, label='QHM lr=1', linestyle='--', color=colors(2))
ax.plot(x, san1_train_loss, label='QHN lr=1', color=colors(2))

ax.plot(x, si_train_loss, label='QHM lr=1.2', linestyle='--', color=colors(3))
ax.plot(x, si1_train_loss, label='QHN lr=1.2', color=colors(3))

ax.plot(x, wu_train_loss, label='QHM lr=1.5', linestyle='--', color=colors(4))
ax.plot(x, wu1_train_loss, label='QHN lr=1.5', color=colors(4))

# ax.plot(x, liu_train_loss, label='QHM β=0.99 v=0.95', linestyle='--', color=colors(5))
# ax.plot(x, liu1_train_loss, label='QHN β=0.99 v=0.95', color=colors(5))
# ax.set_title('lr=1, Train Loss ~ Train Epoch')  # 设置标题
ax.set_xlabel('Epoch')  # 设置 x 轴名字
ax.set_ylabel('Train Loss')	 # 设置 y 轴名字
ax.legend(loc="upper left")  # 调整图例位置
ax.grid(True)  # 添加网格线
# 绘制子图
axins = inset_axes(ax, width="50%", height="40%", loc='center',
                   bbox_to_anchor=(0.3, 0, 1, 1),
                   bbox_transform=ax.transAxes)

axins.plot(x, yi_train_loss, linestyle='--', color=colors(0))
axins.plot(x, yi1_train_loss, label='QHN lr=1.5', color=colors(0))

axins.plot(x, er_train_loss, linestyle='--', color=colors(1))
axins.plot(x, er1_train_loss, color=colors(1))

axins.plot(x, san_train_loss, linestyle='--', color=colors(2))
axins.plot(x, san1_train_loss, color=colors(2))

axins.plot(x, si_train_loss, linestyle='--', color=colors(3))
axins.plot(x, si1_train_loss, color=colors(3))

axins.plot(x, wu_train_loss, linestyle='--', color=colors(4))
axins.plot(x, wu1_train_loss, color=colors(4))

# axins.plot(x, liu_train_loss, label='QHM β=0.999 v=0.99', linestyle='--', color=colors(5))
# axins.plot(x, liu1_train_loss, label='QHN β=0.999 v=0.99', color=colors(5))
# axins.grid(True)  # 添加网格线
# 设置放大区间
zone_left = 120
zone_right = 149
# 坐标轴的扩展比例（根据实际数据调整）
x_ratio = 0  # x轴显示范围的扩展比例
y_ratio = 0.1  # y轴显示范围的扩展比例
# X轴的显示范围
xlim0 = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio
xlim1 = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio
# Y轴的显示范围
y = np.hstack((
               yi_train_loss[zone_left:zone_right],
               yi1_train_loss[zone_left:zone_right],
               er_train_loss[zone_left:zone_right],
               er1_train_loss[zone_left:zone_right],
               san_train_loss[zone_left:zone_right],
               san1_train_loss[zone_left:zone_right],
               si_train_loss[zone_left:zone_right],
               si1_train_loss[zone_left:zone_right],
               wu_train_loss[zone_left:zone_right],
               wu1_train_loss[zone_left:zone_right],
               # liu_train_loss[zone_left:zone_right],
               # liu1_train_loss[zone_left:zone_right],
))
ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio
# 调整子坐标系的显示范围
axins.set_xlim(xlim0, xlim1)
axins.set_ylim(ylim0, ylim1)
# 原图中画方框
tx0 = xlim0
tx1 = xlim1
ty0 = ylim0
ty1 = ylim1
sx = [tx0, tx1, tx1, tx0, tx0]
sy = [ty0, ty0, ty1, ty1, ty0]
ax.plot(sx, sy, "black")
# 画两条线
xy = (xlim0, ylim1)
xy2 = (xlim0, ylim0)
con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                      axesA=axins, axesB=ax)
axins.add_artist(con)
xy = (xlim1, ylim1)
xy2 = (xlim1, ylim0)
con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                      axesA=axins, axesB=ax)
axins.add_artist(con)
# 画图
plt.show()
