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


# 从str获取信息的函数  分为数据与标签
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
z = []  # 存放K个聚类中心


while True:  #随机选择聚类中心
    newz = random.choice(Iris_data_list)
    same_flag = False
    for i in z:
        if (i == newz).all():
            same_flag = True
    if not same_flag:
        z.append(newz)
    if len(z) == K:
        break
assert len(z) == K
print("初始聚类中心:", z)
# 迭代
iterations_num = 0  # 迭代次数计数器
while True:
    z_old = copy.copy(z)
    clusters = [[] for i in range(K)]  # K个聚类蔟中包含的点
    for i in range(len(Iris_data_list)):
        distance = []  # 该点到K个中心的距离表
        for j in range(K):
            distance.append(lg.norm(Iris_data_list[i] - z[j]))
        nearest_zpoint = distance.index(min(distance))  # 找出距离其最近的聚类中心
        clusters[nearest_zpoint].append(i)
    same_flag = True
    for i in range(K):
        cluster_mean = np.zeros(len(z[i]))
        for j in range(len(clusters[i])):
            cluster_mean += Iris_data_list[clusters[i][j]]/len(clusters[i])
        z[i] = cluster_mean
        if (z[i] != z_old[i]).all():
            same_flag = False
    iterations_num += 1
    print(iterations_num, "次迭代后聚类中心:", '\n', z[0], '\n', z[1], '\n', z[2])
    # 计算分类准确率
    clusters_labels = []
    accuracy = 0
    for i in range(K):
        label_list = []
        for j in range(len(clusters[i])):
            label_list.append(Iris_label_list[clusters[i][j]])
        true_label = []
        for j in range(K):
            true_label.append(label_list.count(j))
        accuracy += max(true_label)  # 选取数量最大的标签作为其标签
        print(true_label.index(max(true_label)), end=' ')
        clusters_labels.append(true_label.index(max(true_label)))
    accuracy = accuracy/len(Iris_label_list)
    print("分类准确率为:", accuracy)
    if same_flag:
        if len(set(clusters_labels)) == K:
            print('已找到聚类结果')
        else:
            print('聚类结果错误！')
        break
