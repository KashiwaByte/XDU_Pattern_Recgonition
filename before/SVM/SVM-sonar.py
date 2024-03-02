import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxopt
import random
from sklearn.metrics import accuracy_score


# 数据标准化
def generalization(x):
    num_feature = x.shape[1]
    for i in range(num_feature):
        x[:, i] = (x[:, i] - min(x[:, i])) / (max(x[:, 1]) - min(x[:, i]))
    return x


# 训练集测试集分割
def train_test_split(x, y, test_rate):
    num_x = x.shape[0]
    # 打乱
    index = list(range(num_x))
    random.shuffle(index)
    x = x[index]
    y = y[index]

    # 分割
    split = round(num_x * test_rate)
    x_test = x[:split, :]
    x_train = x[split:, :]
    y_test = y[:split]
    y_train = y[split:]
    return x_train, x_test, y_train, y_test

# Sonar数据集读取
# 208个样本，60维特征（不同角度返回的值），二分类结果为岩石/金属
data2 = pd.read_csv('.\data\sonar.all-data', header=None)
data2 = np.array(data2)
x2 = np.float64(data2[:, 0:60])
y2 = data2[:, 60]
for i in range(len(y2)):
    if y2[i] == 'R':
        y2[i] = -1
    else:
        y2[i] = 1
y2 = np.int64(y2)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_rate=0.3)


# 定义三种常见的核函数
# 线性核
def linear_kernal(**kwargs):
    def K(x1, x2):
        return np.inner(x1, x2)

    return K


# 多项式核
def poly_kernal(power, coef, **kwargs):
    def K(x1, x2):
        return (np.inner(x1, x2) + coef) ** power

    return K


# 径向基核
def rbf_kernal(gamma, **kwargs):
    def K(x1, x2):
        distance = np.linalg.norm(x1 - x2) ** 2
        return np.exp(-gamma * distance)

    return K


class svm:
    # 输入参数
    # param kernal:核函数
    # param penaltyC:软间隔惩罚项C
    # param power, gamma,coef为核函数的一些参数
    def __init__(self, kernal=linear_kernal, penaltyC=1, power=1, gamma=1, coef=1):
        self.kernal = kernal
        self.penaltyC = penaltyC
        self.power = power
        self.gamma = gamma
        self.coef = coef
        self.kernal = self.kernal(
            power=self.power,
            gamma=self.gamma,
            coef=self.coef)

    def train(self, x, y):
        x_num = x.shape[0]
        kernal_matrix = self.kernal(x, x) + (1 / self.penaltyC) * np.eye(x_num)

        # 计算标准凸二次规划的几个参数
        p = cvxopt.matrix(kernal_matrix * np.outer(y, y))
        q = cvxopt.matrix(-np.ones([x_num, 1], np.float64))
        g = cvxopt.matrix(-np.eye(x_num))
        h = cvxopt.matrix(np.zeros([x_num, 1], np.float64))

        y = np.float64(y)
        a = cvxopt.matrix(y, (1, x_num))
        b = cvxopt.matrix(0.)

        # 使用凸规划工具包cvxopt求解SVM目标函数（算lagrange乘子）
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(p, q, g, h, a, b)
        alpha = np.float32(np.array(solution['x']))
        alpha[alpha <= 1e-4] = 0

        # 求权重w和截距b
        w = np.sum(np.reshape(y, [-1, 1]) * alpha * x, axis=0)
        b = np.mean(np.reshape(y, [-1, 1]) - np.reshape(np.dot(w, np.transpose(x)), [-1, 1]))
        self.w = w
        self.b = b

        return w, b, alpha

    def predict(self, x_test):
        y_predict = []
        for sample in x_test:
            predict1 = self.kernal(self.w, sample) + self.b
            predict1 = np.int64(np.sign(predict1))
            y_predict.append(predict1)
        #         y_predict.tolist()
        return y_predict


if __name__ == '__main__':

    a = []
    for i in range(1,11):
        svm2 = svm(poly_kernal, penaltyC=i*i)
        w2, b2, alpha2 = svm2.train(x2, y2)
        y2_predict = svm2.predict(x2_test)
        print('C=',i,' Sonar_accuracy:', accuracy_score(y2_test, y2_predict))
        a.append(accuracy_score(y2_test, y2_predict))

plt.figure(1)
plt.plot(range(1,11),a)
plt.grid()
plt.xlabel('penaltyC')
plt.ylabel('acc')
plt.show()
