# -*- coding: utf-8 -*-
# Author：KashiwaByte
"""

数据集大小分别为 (7291, 256) (7291, 1) (2007, 256) (2007, 1)
"""
import h5py
import pandas as pd
import numpy as np
import swanlab
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)
swanlab.init(experiment_name="KNN_Usps_GPU")
# 读取 USPS数据集
def pre_handle():
    with h5py.File('data/usps.h5') as hf:
            train = hf.get('train')
            x_train = train.get('data')[:]
            y_train = train.get('target')[:]
            test = hf.get('test')
            x_test = test.get('data')[:]
            y_test = test.get('target')[:]

    train_data=pd.DataFrame(x_train)
    train_label=pd.DataFrame(y_train)
    test_data=pd.DataFrame(x_test)
    test_label=pd.DataFrame(y_test)
    return train_data,train_label,test_data,test_label


train_data,train_label,test_data,test_label = pre_handle()


train_data = np.mat(train_data)
train_label = np.mat(train_label)
test_data = np.mat(test_data)
test_label = np.mat(test_label)
train_label = train_label.astype(int)
test_label = test_label.astype(int)

train_data_tensor = torch.from_numpy(train_data)
train_label_tensor = torch.from_numpy(train_label)
test_data_tensor = torch.from_numpy(test_data)
test_label_tensor = torch.from_numpy(test_label)

train_data_tensor = train_data_tensor.to(device)
train_label_tensor = train_label_tensor.to(device)
test_data_tensor = test_data_tensor.to(device)
test_label_tensor = test_label_tensor.to(device)


def k_nn(train_data_tensor,train_label_tensor,test_data_tensor,test_label_tensor):
    accuracy = 0
    for i in range(2007):
        count = torch.zeros(10,device=device)
        prediction = 0
        distance = torch.zeros((7291,2),device=device)
        for t in range(7291):
            distance[t,1] = torch.norm(test_data_tensor[i]-train_data_tensor[t])
            distance[t,0] = train_label_tensor[t]           # 储存标签和欧式距离
       # order = distance[np.lexsort(distance.T)]    # 按最后一列排序
        # 对最后一列进行排序，并获取排序后的索引
        sorted_indices = torch.argsort(distance[:, -1])

        # 使用这些索引来重新排序整个distances张量
        order = distance[sorted_indices]
        
        for n in range(k):
            a = order[n,0]
            a = int(a)
            count[a] += 1   
        prediction = count.argmax()                           # 取出现次数最多的为预测值
        if prediction == test_label_tensor[i]:
            accuracy += 1
    Accuracy = accuracy/2007
    swanlab.log({"Accurary":Accuracy})
    if k==1:
        print(f"k={k}时，Iris数据集的最近邻准确率为：",Accuracy)
    else:
        print(f"k={k}时，Iris数据集的k近邻准确率为：",Accuracy)
   # print("USPS数据集的最近邻准确率为:",Accuracy)
    return Accuracy

Res = torch.zeros(20,device=device)

for m in range(2):
    k = m+1
    Res[m] = k_nn(train_data_tensor,train_label_tensor,test_data_tensor,test_label_tensor)


# 绘制 k与分类准确率的图像
import matplotlib.pyplot as plt

x = torch.arange(1,21,1,device=device)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.ylim((0.5,1))            # y坐标的范围
#画图
plt.plot(x,Res,'r')
swanlab.log({"KNN_Usps_GPU":swanlab.Image(plt)})
plt.savefig("result/KNN_Usps_GPU.jpg",dpi=2000)















