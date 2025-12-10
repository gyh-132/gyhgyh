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


seed = 56  # 24, 32, 42, 48, 56, 64, 128, 512, 1024, 2048
A, B, c, f_min, X_min, L = generate_parameters(seed)
# A, B, c, f_min, X_min, L = np.array([[1, 0], [0, 10]]), np.array([0, 0]), 0, 0, np.array([0, 0]), 10
print("A =", A)
print("B =", B)
print("c =", c)
print("Minimum value =", f_min)
print("Minimizer X_min =", X_min)
print("Lipschitz constant L =", L)

ran = 20
kb = 1
k_list = [(i+1)/((kb+0)*L) for i in range(kb)]
# k_list = [2/(5*L), 4/(5*L)]
CM_data = [[] for _ in range(len(k_list))]
NAG_data = [[] for _ in range(len(k_list))]
QHM1_data = [[] for _ in range(len(k_list))]
QHM2_data = [[] for _ in range(len(k_list))]
QHN1_data = [[] for _ in range(len(k_list))]
QHN2_data = [[] for _ in range(len(k_list))]

# 算法迭代
for i in range(len(k_list)):
    k = k_list[i]

    # CM
    x = X_min + np.array([1, 1])
    t = 0
    m = 0
    CM_data[i].append(x)
    for _ in range(ran):
        t = t + 1
        lr = k * (3 * t + 1) / 12
        v = 1
        beta = 0.9

        eva, der = evaluation_derivation(A, B, c, x)
        # print(f"CM({i+1}) 第{t}次迭代，当前x为{x}，函数值为{eva}，与最小值的差为{eva - f_min}")
        if abs(eva - f_min) <= 1e-2:
            break

        m = (1 - beta) * der + beta * m
        x = x - lr * ((1 - v) * der + (v * m))
        CM_data[i].append(x)
    eva, der = evaluation_derivation(A, B, c, x)

    # NAG
    x = X_min + np.array([1, 1])
    t = 0
    m = 0
    NAG_data[i].append(x)
    for _ in range(ran):
        t = t + 1
        lr = k * (3 * t + 1) / 12
        v = 1
        beta = 0.9

        eva, _ = evaluation_derivation(A, B, c, x)
        _, der = evaluation_derivation(A, B, c, x - (lr * v * beta * m))
        # print(f"NAG({i+1}) 第{t}次迭代，当前x为{x}，函数值为{eva}，与最小值的差为{eva - f_min}")
        if abs(eva - f_min) <= 1e-2:
            break

        m = (1 - beta) * der + beta * m
        x = x - lr * ((1 - v) * der + (v * m))
        NAG_data[i].append(x)
    eva, der = evaluation_derivation(A, B, c, x)

    # QHM(v=beta=0.9)
    x = X_min + np.array([1, 1])
    t = 0
    m = 0
    QHM1_data[i].append(x)
    for _ in range(ran):
        t = t + 1
        lr = k * (3 * t + 1) / 12
        v = 0.9
        beta = 0.9

        eva, der = evaluation_derivation(A, B, c, x)
        # print(f"QHM({i+1}) 第{t}次迭代，当前x为{x}，函数值为{eva}，与最小值的差为{eva - f_min}")
        if abs(eva - f_min) <= 1e-2:
            break

        m = (1 - beta) * der + beta * m
        x = x - lr * ((1 - v) * der + (v * m))
        QHM1_data[i].append(x)
    eva, der = evaluation_derivation(A, B, c, x)

    # QHM(v=0.7, beta=0.999)
    x = X_min + np.array([1, 1])
    t = 0
    m = 0
    QHM2_data[i].append(x)
    for _ in range(ran):
        t = t + 1
        lr = k * (3 * t + 1) / 12
        v = 0.7
        beta = 0.999

        eva, der = evaluation_derivation(A, B, c, x)
        # print(f"QHM({i+1}) 第{t}次迭代，当前x为{x}，函数值为{eva}，与最小值的差为{eva - f_min}")
        if abs(eva - f_min) <= 1e-2:
            break

        m = (1 - beta) * der + beta * m
        x = x - lr * ((1 - v) * der + (v * m))
        QHM2_data[i].append(x)
    eva, der = evaluation_derivation(A, B, c, x)

    # QHN(v=beta=0.9)
    x = X_min + np.array([1, 1])
    t = 0
    m = 0
    QHN1_data[i].append(x)
    for _ in range(ran):
        t = t + 1
        lr = k * (3 * t + 1) / 12
        v = 0.9
        beta = 0.9

        eva, _ = evaluation_derivation(A, B, c, x)
        _, der = evaluation_derivation(A, B, c, x - (lr * v * beta * m))
        # print(f"QHN({i+1}) 第{t}次迭代，当前x为{x}，函数值为{eva}，与最小值的差为{eva - f_min}")
        if abs(eva - f_min) <= 1e-2:
            break

        m = (1 - beta) * der + beta * m
        x = x - lr * ((1 - v) * der + (v * m))
        QHN1_data[i].append(x)
    eva, _ = evaluation_derivation(A, B, c, x)

    # QHN(v=0.7, beta=0.999)
    x = X_min + np.array([1, 1])
    t = 0
    m = 0
    QHN2_data[i].append(x)
    for _ in range(ran):
        t = t + 1
        lr = k * (3 * t + 1) / 12
        v = 0.7
        beta = 0.999

        eva, _ = evaluation_derivation(A, B, c, x)
        _, der = evaluation_derivation(A, B, c, x - (lr * v * beta * m))
        # print(f"QHN({i+1}) 第{t}次迭代，当前x为{x}，函数值为{eva}，与最小值的差为{eva - f_min}")
        if abs(eva - f_min) <= 1e-2:
            break

        m = (1 - beta) * der + beta * m
        x = x - lr * ((1 - v) * der + (v * m))
        QHN2_data[i].append(x)
    eva, _ = evaluation_derivation(A, B, c, x)


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


# 为每个k值绘制单独的子图
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# 绘制等值线
levels = np.linspace(f_min, f_min + 10, 10)  # 等值线范围
contour = ax.contour(X1, X2, Z, levels=levels, cmap='coolwarm', alpha=0.6)
plt.colorbar(contour, ax=ax, label='value')

# 标记最小值点
ax.plot(X_min[0], X_min[1], 'r*', markersize=12, label='Minimum point')

# 提取轨迹数据
CM_traj = np.array(CM_data[0])
NAG_traj = np.array(NAG_data[0])
QHM1_traj = np.array(QHM1_data[0])
QHM2_traj = np.array(QHM2_data[0])
QHN1_traj = np.array(QHN1_data[0])
QHN2_traj = np.array(QHN2_data[0])

ax.plot(CM_traj[:, 0], CM_traj[:, 1], 'y--o',
        markersize=3, linewidth=1, label=f'CM (β=0.9)')
ax.plot(NAG_traj[:, 0], NAG_traj[:, 1], 'b--s',
        markersize=3, linewidth=1, label=f'NAG (β=0.9)')
# ax.plot(QHM1_traj[:, 0], QHM1_traj[:, 1], 'r--^',
#         markersize=3, linewidth=1, label=f'QHM (v=β=0.9)')
ax.plot(QHM2_traj[:, 0], QHM2_traj[:, 1], 'r--^',
        markersize=3, linewidth=1, label=f'QHM (v=0.7, β=0.999)')
# ax.plot(QHN1_traj[:, 0], QHN1_traj[:, 1], 'g--D',
#         markersize=3, linewidth=1, label=f'QHN (v=β=0.9)')
ax.plot(QHN2_traj[:, 0], QHN2_traj[:, 1], 'g--D',
        markersize=3, linewidth=1, label=f'QHN (v=0.7, β=0.999)')

# # 标记起点
ax.plot(CM_traj[0, 0], CM_traj[0, 1], 'bo',
        markersize=8, markerfacecolor='none', label=f'Initial point')

# 设置标题和标签
ax.set_title(f'L = {L:.3f}, η=(3t+1)/12L', fontsize=14)
ax.set_xlabel('x1', fontsize=14)
ax.set_ylabel('x2', fontsize=14)
ax.legend(fontsize=14)
ax.grid(True)
plt.tight_layout()
plt.show()












