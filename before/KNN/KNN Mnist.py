# 调用库
import os
import struct
import numpy as np
import heapq
import time

# 数据读取函数


# 读取数据集的函数,返回图片的像素点灰度值数据和数字标签
# 读取训练集数据
def load_mnist_train(path, kind='train'):
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


# 读取测试集数据
def load_mnist_test(path, kind='t10k'):
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


# 读取数据，打印数据属性
# 读取数据
path = r"MNIST"
train_set, train_labels = load_mnist_train(path)  # 训练集
test_set, test_labels = load_mnist_test(path)  # 测试集

print(train_set.shape)
print(train_labels.shape)
print(test_set.shape)
print(test_labels.shape)
train_size = train_set.shape[0]  # 训练样本数
test_size = test_set.shape[0]  # 测试样本数
"""
outputs:
(60000, 784) # 784是28 * 28,每一个图片中的像素点数，共有60000个训练集
(60000,)
(10000, 784) # 10000个测试集
(10000,)
"""
# 数据归一化
num_pixel = 784  # 记录像素点数
# 这里采用策略归一化到[0.0, 1.0]区间内
train_set = train_set / 255.0
test_set = test_set / 255.0
# 采用部分数据进行测试，减少计算时间成本，检验合理性，验证算法是否正常工作
# 先选择部分数据集进行测试
num_chosen_train = 12000  # 选用12000个训练数据
num_chosen_test = 2000  # 选用2000个测试数据

# 随机选择下标
train_chosen_index = np.random.choice(np.arange(train_size), size=num_chosen_train, replace=False)
test_chosen_index = np.random.choice(np.arange(test_size), size=num_chosen_test, replace=False)

# 随机选择出的训练图片
train_part_images = train_set[train_chosen_index]
train_part_labels = train_labels[train_chosen_index]

# 随机选择出的测试图片
test_part_images = test_set[test_chosen_index]
test_part_labels = test_labels[test_chosen_index]
# 创建KNNClassifier类
"""
为了充分利用numpy对矩阵计算的优化，将每一个测试集的1 * 784的数据堆叠为len(x_train) * 784的矩阵，
然后进行直接相减求二次幂，每一行的结果相加的方式更加快速地求得测试集与每一个训练集的欧几里得距离的平方大小；
（不执行开根运算效果一样，减少运算量）
"""


# 分类器类实现
class KNNClassifier():
    def __init__(self, k=10):
        self.k = k
        self.x_train = None
        self.y_train = None

    # 传入训练数据
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        labels_pred = np.zeros(len(x_test))  # 测试集待预测结果
        for i in range(len(x_test)):
            # np.tile()对单个测试集数据进行堆叠，但是堆叠是在同一维度上
            # 使用reshape将一维堆叠结果转化为与self.x_train大小相同的矩阵
            X = np.reshape(np.tile(x_test[i], self.x_train.shape[0]), (self.x_train.shape[0], num_pixel))
            distance = np.sum(np.square(self.x_train - X), axis=1)  # 使用欧氏距离的平方作为“距离”度量
            index = heapq.nsmallest(self.k, range(len(distance)), distance.take)  # 找到distance中最小的k个值的下标
            label = self.y_train[index]  # 待选的k个label
            labels_pred[i] = np.argmax(np.bincount(label))  # bincount()实现对标签出现次数的计算，比如5出现3次，那么返回np数组就有下标5对应数据3
        return labels_pred

    def score(self, y_pred, y_test):
        rate = np.sum(np.array(y_pred == y_test, dtype=np.int32))
        return rate / len(y_test)  # / test_length


# 调用类进行测试
# 用不同的k值进行测试
k_range = range(1, 11)
accuracy = []  # 准确率
# start_t = time.time() # 计算运行时间
for k in k_range:
    classifier = KNNClassifier(k=k)  # 对于不同的k值创建classifier
    classifier.fit(train_part_images, train_part_labels)
    pred = classifier.predict(test_part_images)
    result = classifier.score(pred, test_part_labels)  # score-->accuracy
    accuracy.append(result)
    print("accuracy:", result)
# end_t = time.time()
# print("total time:", end_t - start_t)

"""
outputs:
accuracy: 0.9515
accuracy: 0.9415
accuracy: 0.9525
accuracy: 0.952
accuracy: 0.952
accuracy: 0.9475
accuracy: 0.9515
accuracy: 0.9485
accuracy: 0.9455
accuracy: 0.946
"""
# 准确率 - K值图
import matplotlib.pyplot as plt

plt.plot(k_range, accuracy)
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.show()