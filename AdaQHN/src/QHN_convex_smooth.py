import numpy as np
import matplotlib.pyplot as plt


def generate_parameters(seed=0):
    np.random.seed(seed)
    # Generate A as a positive definite matrix
    while True:
        A_a = np.random.randint(-5, 6)
        A_b = np.random.randint(-5, 6)
        A_c = np.random.randint(1, 11)
        A = np.array([[A_a*A_a, A_a*A_b], [A_a*A_b, A_b*A_b + A_c]])
        if A_a != 0:
            break

    # Generate B and c
    B = np.random.randint(-5, 6, size=(2,))
    c = np.random.randint(-5, 6)

    # Compute the minimizer X_min = -0.5 * A^{-1} B
    A_inv = np.linalg.inv(A)
    X_min = -0.5 * A_inv @ B

    # Compute the minimum value f_min = X_min^T A X_min + B^T X_min + c
    f_min = X_min.T @ A @ X_min + B.T @ X_min + c

    # Compute the Lipschitz constant L = 2 * spectral norm of A, then round up
    eigenvalues = np.linalg.eigvals(A)
    L = 2 * np.max(eigenvalues)
    # L = 10 * int(np.ceil(L / 10))  # Round up to the nearest multiple of 10

    return A, B, c, f_min, X_min, L


def evaluation_derivation(A, B, c, x=np.zeros((2,))):
    eva = x.T @ A @ x + B.T @ x + c
    der = 2 * A @ x + B
    return eva, der


seed = 1024
A, B, c, f_min, X_min, L = generate_parameters(seed)
# A, B, c, f_min, X_min, L = np.array([[1, 0], [0, 10]]), np.array([0, 0]), 0, 0, np.array([0, 0]), 10
print("A =", A)
print("B =", B)
print("c =", c)
print("Minimum value =", f_min)
print("Minimizer X_min =", X_min)
print("Lipschitz constant L =", L)

ran = 50
kb = 1
k_list = [(i+1)/((kb+0)*L) for i in range(kb)]
# k_list = [2/(5*L), 4/(5*L)]
Momentum_data = [[] for _ in range(len(k_list))]
QHM_data = [[] for _ in range(len(k_list))]
QHN_data = [[] for _ in range(len(k_list))]

# 算法迭代
for i in range(len(k_list)):
    k = k_list[i]

    # Momentum
    x = X_min + np.array([1, 1])
    t = 0
    m = 0
    Momentum_data[i].append(x)

    for _ in range(ran):
        t = t + 1
        # lr = k * (3 * t + 1) / 12
        # v = (3 * t - 7) / (3 * t + 5)
        # beta = (t - 1) * (3 * t - 10) / ((t + 2) * (3 * t - 7))
        lr = k * (3 * t + 1) / 12
        v = 0.9
        beta = 0.9

        eva, der = evaluation_derivation(A, B, c, x)
        # print(f"QHM({i+1}) 第{t}次迭代，当前x为{x}，函数值为{eva}，与最小值的差为{eva - f_min}")
        if abs(eva - f_min) <= 1e-2:
            break

        m = (1 - beta) * der + beta * m
        x = x - lr * ((1 - v) * der + (v * m))
        Momentum_data[i].append(x)
    eva, der = evaluation_derivation(A, B, c, x)

    # QHM
    x = X_min + np.array([1, 1])
    t = 0
    m = 0
    QHM_data[i].append(x)

    for _ in range(ran):
        t = t + 1
        # lr = k * (3 * t + 1) / 12
        # v = (3 * t - 7) / (3 * t + 5)
        # beta = (t - 1) * (3 * t - 10) / ((t + 2) * (3 * t - 7))
        lr = k * (3 * t + 1) / 12
        v = 1
        beta = 0.9

        # eva, der = evaluation_derivation(A, B, c, x)
        eva, _ = evaluation_derivation(A, B, c, x)
        _, der = evaluation_derivation(A, B, c, x - (lr * v * beta * m))
        # print(f"QHM({i+1}) 第{t}次迭代，当前x为{x}，函数值为{eva}，与最小值的差为{eva - f_min}")
        if abs(eva - f_min) <= 1e-2:
            break

        m = (1 - beta) * der + beta * m
        x = x - lr * ((1 - v) * der + (v * m))
        QHM_data[i].append(x)
    eva, der = evaluation_derivation(A, B, c, x)
    print(f"QHM({i+1}) 迭代{t}次，最终x为{x}，函数值为{eva}，与最小值的差为{eva-f_min}")

    # QHN
    x = X_min + np.array([1, 1])
    t = 0
    m = 0
    QHN_data[i].append(x)

    for _ in range(ran):
        t = t + 1
        lr = k * (3 * t + 1) / 12
        # v = (t * (3 * t + 1) - 12) / (t * (3 * t + 1))
        # beta = t * (t * (3 * t - 5) - 10) / ((t + 2) * (t * (3 * t + 1) - 12))
        v = 0.9
        beta = 0.9

        eva, _ = evaluation_derivation(A, B, c, x)
        _, der = evaluation_derivation(A, B, c, x - (lr * v * beta * m))
        # print(f"QHN({i+1}) 第{t}次迭代，当前x为{x}，函数值为{eva}，与最小值的差为{eva - f_min}")
        if abs(eva - f_min) <= 1e-2:
            break

        m = (1 - beta) * der + beta * m
        x = x - lr * ((1 - v) * der + (v * m))
        QHN_data[i].append(x)
    eva, _ = evaluation_derivation(A, B, c, x)
    print(f"QHN({i + 1}) 迭代{t}次，最终x为{x}，函数值为{eva}，与最小值的差为{eva - f_min}")


# 绘图
# 生成等值线网格
x1 = np.linspace(X_min[0] - 1.5, X_min[0] + 1.5, 100)
x2 = np.linspace(X_min[1] - 1.5, X_min[1] + 1.5, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = np.zeros_like(X1)
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        x = np.array([X1[i, j], X2[i, j]])
        Z[i, j] = x.T @ A @ x + B.T @ x + c  # 计算二次函数值

# # 为每个k值绘制单独的子图
# fig, axes = plt.subplots(1, kb, figsize=(18, 6))
#
# for i, k in enumerate(k_list):
#     ax = axes[i]
#
#     # 绘制等值线
#     levels = np.linspace(f_min, f_min + 2, 10)  # 等值线范围
#     contour = ax.contour(X1, X2, Z, levels=levels, cmap='coolwarm', alpha=0.6)
#     plt.colorbar(contour, ax=ax, label='value')
#
#     # 标记最小值点
#     ax.plot(X_min[0], X_min[1], 'r*', markersize=12, label='Minimum point')
#
#     # 提取轨迹数据
#     QHM_traj = np.array(QHM_data[i])
#     QHN_traj = np.array(QHN_data[i])
#
#     # 绘制QHM和QHN轨迹
#     ax.plot(QHM_traj[:, 0], QHM_traj[:, 1], 'g--s',
#             markersize=3, linewidth=1, label=f'QHM ((k={k:.3f}))')
#     ax.plot(QHN_traj[:, 0], QHN_traj[:, 1], 'b-o',
#             markersize=3, linewidth=1, label=f'QHN ((k={k:.3f}))')
#
#     # # 标记起点
#     ax.plot(QHM_traj[0, 0], QHM_traj[0, 1], 'bo',
#             markersize=8, markerfacecolor='none', label=f'X0 = ({QHM_traj[0]})')
#     # ax.plot(QHM_traj[0, 0], QHM_traj[0, 1], 'bo',
#     #         markersize=8, markerfacecolor='none', label='QHM起点')
#     # ax.plot(QHN_traj[0, 0], QHN_traj[0, 1], 'gs',
#     #         markersize=8, markerfacecolor='none', label='QHN起点')
#
#     # 设置标题和标签
#     ax.set_title(f'k = {k:.3f}, Lipschitz L = {L:.3f}')
#     ax.set_xlabel('x1')
#     ax.set_ylabel('x2')
#     ax.legend()
#     ax.grid(True)
#
#     # # 自动调整坐标轴范围
#     # all_points = np.vstack((QHM_traj, QHN_traj))
#     # x_pad = (all_points[:, 0].max() - all_points[:, 0].min()) * 0.2
#     # y_pad = (all_points[:, 1].max() - all_points[:, 1].min()) * 0.2
#     # ax.set_xlim(all_points[:, 0].min() - x_pad, all_points[:, 0].max() + x_pad)
#     # ax.set_ylim(all_points[:, 1].min() - y_pad, all_points[:, 1].max() + y_pad)
#
# plt.tight_layout()
# plt.show()


# 为每个k值绘制单独的子图
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# 绘制等值线
levels = np.linspace(f_min, f_min + 10, 10)  # 等值线范围
contour = ax.contour(X1, X2, Z, levels=levels, cmap='coolwarm', alpha=0.6)
plt.colorbar(contour, ax=ax, label='value')

# 标记最小值点
ax.plot(X_min[0], X_min[1], 'r*', markersize=12, label='Minimum point')

# 提取轨迹数据
Momentum_traj = np.array(Momentum_data[0])
QHM_traj = np.array(QHM_data[0])
QHN_traj = np.array(QHN_data[0])

ax.plot(Momentum_traj[:, 0], Momentum_traj[:, 1], 'r--s',
        markersize=3, linewidth=1, label=f'QHM (v=β=0.9, η=(3t+1)/12L)')
ax.plot(QHM_traj[:, 0], QHM_traj[:, 1], 'g--s',
        markersize=3, linewidth=1, label=f'NAG (β=0.9, η=(3t+1)/12L)')
ax.plot(QHN_traj[:, 0], QHN_traj[:, 1], 'b-o',
        markersize=3, linewidth=1, label=f'QHN (v=β=0.9, η=(3t+1)/12L)')

# # 标记起点
ax.plot(QHM_traj[0, 0], QHM_traj[0, 1], 'bo',
        markersize=8, markerfacecolor='none', label=f'X0 = ({QHM_traj[0]})')
# ax.plot(QHM_traj[0, 0], QHM_traj[0, 1], 'bo',
#         markersize=8, markerfacecolor='none', label='QHM起点')
# ax.plot(QHN_traj[0, 0], QHN_traj[0, 1], 'gs',
#         markersize=8, markerfacecolor='none', label='QHN起点')

# 设置标题和标签
ax.set_title(f'Lipschitz L = {L:.3f}')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.legend()
ax.grid(True)


plt.tight_layout()
plt.show()












