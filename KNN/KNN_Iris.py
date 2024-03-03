# -*- coding: utf-8 -*-
# Author：KashiwaByte

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import datasets
import swanlab

iris = pd.read_csv('data/iris.data',header=None,sep=',')

swanlab.init(experiment_name="KNN_Iris")



def k_nn(X):
    accuracy = 0
    for i in range(150):
        count1 = 0
        count2 = 0
        count3 = 0
        prediction = 0
        distance = np.zeros((149,2))
        test = X[i]
        train = np.delete(X,i,axis=0)
        test1 = test[:,0:4]
        train1 = train[:,0:4]
        for t in range(149):
            distance[t,1] = np.linalg.norm(test1 - train1[t])
            distance[t,0] = train[t,4]              # 储存标签和欧式距离
        order = distance[np.lexsort(distance.T)]    # 按最后一列排序
        
        for n in range(k):
            if order[n,0] == 1:
                count1 +=1
            if order[n,0] == 2:
                count2 +=1
            if order[n,0] == 3:
                count3 +=1
        if count1 >= count2 and count1 >= count3:
            prediction = 1
        if count2 >= count1 and count2 >= count3:
            prediction = 2
        if count3 >= count1 and count3 >= count2:
            prediction = 3                         # 取出现次数最多的为预测值
        if prediction == test[0,4]:
            accuracy += 1
    Accuracy = accuracy/150
    swanlab.log({"Accuracy":Accuracy})
    if k==1:
        print(f"k={k}时，Iris数据集的最近邻准确率为：",Accuracy)
    else:
        print(f"k={k}时，Iris数据集的k近邻准确率为：",Accuracy)
    return Accuracy


    


x = iris.iloc[:,0:4]
x = np.mat(x)
a = np.full((50,1),1)
b = np.full((50,1),2)
c = np.full((50,1),3)
Res = np.zeros(50)

d = np.append(a,b,axis=0)
d = np.append(d,c,axis=0)
X = np.append(x,d,axis=1)         # 将数据集中的标签更换为1，2，3


for m in range(50):
    k = m + 1 
    Res[m] = k_nn(X)


# 绘制 k与分类准确率的图像
import matplotlib.pyplot as plt

x = np.arange(1,51,1)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.ylim((0.8,1))            # y坐标的范围
#画图
plt.plot(x,Res,'b')
swanlab.log({"result":swanlab.Image(plt)})
plt.savefig("result/k近邻_Iris.jpg",dpi=2000)
























