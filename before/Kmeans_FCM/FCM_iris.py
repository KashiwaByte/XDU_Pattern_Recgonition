import numpy as np
import numpy.linalg as lg
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import copy

# 读入文件
path = r'iris.data'
f = open(path, 'r')
data = f.read()
data = data.split()

# 从str获取信息的函数
def get_data(datastr):
    species = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    datastr = datastr.split(',')
    data_part = list(map(float, datastr[:4]))
    label_part = species[datastr[4]]

    data_part = np.array(data_part)
    return data_part, label_part

# 创建样本空间向量集合
Iris_data_list = []
Iris_label_list = []
for iris in data:
    iris_data, iris_label = get_data(iris)
    Iris_data_list.append(iris_data)
    Iris_label_list.append(iris_label)

# 初始化K个聚类中心
K = 3  # 聚类中心个数
z = [0, 0, 0]  # 存放K个聚类中心

# 初始化隶属矩阵U
U = np.random.rand(len(Iris_data_list), K)
for i in range(len(Iris_data_list)):
    U[i] = U[i]/sum(U[i])

# 迭代
J = 0
a = 2  # 柔性参数
iterations_num = 0  # 迭代次数计数器

while True:
    z_old = copy.copy(z)
    U_old = copy.copy(U)
    J_old = J
    # 计算新聚类中心
    for j in range(K):
        sum_ux = 0
        sum_u = 0
        for i in range(len(Iris_data_list)):
            sum_ux += (U[i][j]**a) * Iris_data_list[i]
            sum_u += U[i][j]**a
        z[j] = sum_ux/sum_u
    iterations_num += 1
    print(iterations_num, "次迭代后聚类中心:", '\n', z[0], '\n', z[1], '\n', z[2])

    # 计算代价函数
    J = 0
    for j in range(K):
        for i in range(len(Iris_data_list)):
            J += (U[i][j]**a) * (lg.norm(z[j]-Iris_data_list[i])**2)
    # 终止条件
    if abs(J - J_old) < 0.0001:
        break
    # 计算新矩阵U
    for i in range(len(Iris_data_list)):
        for j in range(K):
            sum_ud = 0
            for k in range(K):
                sum_ud += ((lg.norm(z[j]-Iris_data_list[i])) / (lg.norm(z[k]-Iris_data_list[i]))) ** (2/(a-1))
            U[i][j] = 1/sum_ud

# 计算第几蔟的实际标签是什么
label_order = []
for i in range(K):
    K_list = [0]*K
    for j in range(len(Iris_data_list)):
        if np.argmax(U[j]) == i:
            K_list[Iris_label_list[j]] += 1
    label_order.append(K_list.index(max(K_list)))
assert len(set(label_order)) == K, '出现了两类相同蔟！'
# 输出判断结果
for i in range(K):
    print('第{}蔟的实际标签为:{}'.format(i+1, label_order[i]), end='  ')
print()

# 计算实际标签对应的是第几蔟
un_label_order = [0]*K
for i in range(K):
    un_label_order[label_order[i]] = i
accuracy1 = 0
for i in range(len(Iris_data_list)):
    if U[i][un_label_order[Iris_label_list[i]]] == max(U[i]):
        accuracy1 += 1
accuracy1 /= len(Iris_data_list)
print("归1分类准确率为:", accuracy1)
accuracy2 = 0
for i in range(len(Iris_data_list)):
    accuracy2 += U[i][un_label_order[Iris_label_list[i]]]
accuracy2 /= len(Iris_data_list)
print("模糊分类准确率为:", accuracy2)
