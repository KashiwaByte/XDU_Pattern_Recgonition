from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split


# 函数定义部分

# 定义转换器字典用于转换对应兰花数字（相当于loadtxt函数转换器参数converters的字典）
def Sort_dic(type):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[type]


# 具体实现部分

# 读取数据集的数据并进行简单清洗

path = 'data/Iris.data'
# converters是数据转换器定义，将第5列的花名格式str转化为0,1,2三种数字分别代表不同的类别兰花类别（这是一步数据清洗过程）
data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: Sort_dic})

# 将数据和标签列划分开来

# split函数的参数意义(数据，分割位置（这里用了一种不常见的写法表示前四列为一组记作x，后面剩余部分为一组记作y），
# axis = 1（代表水平分割，以每一个行记录为切割对象） 或 0（代表垂直分割，以属性为切割对象）)。
x, y = np.split(data, indices_or_sections=(4,), axis=1)  # x为数据，y为标签
# 为便于后边画图显示，只选取前两维度。若不用画图，可选取前四列x[:,0:4]就选中所有特征了
x = x[:, 0:2]  # 标记一下，这个切片的意思是提取前两列的每一行[每一行,0,1两列]
# Sklearn库函数train_test_split可以实现将数据集按比例划分为训练集和测试集
train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=1, train_size=0.8, test_size=0.2)
# 目前x为数据y为标签（即标注样本属于哪一类）
Training_set_score=1
Testing_set_score=5

# C越大分类效果越好，但有可能会过拟合，gamma是高斯rbf核参数，而后面的dfs制定了类别划分方式，ovr是一对多方式。
classifier = svm.SVC(C=5, kernel='poly', gamma=20, decision_function_shape='ovr')

classifier.fit(train_data, train_label.ravel())  # 用训练集数据来训练模型。（ravel函数在降维时默认是行序优先）

# 计算svc分类器的准确率

print("Training_set_score：", format(classifier.score(train_data, train_label), '.3f'))
print("Testing_set_score：", format(classifier.score(test_data, test_label), '.3f'))

# 绘制图形将实验结果可视化

# 首先确定坐标轴范围，通过二维坐标最大最小值来确定范围
# 第1维特征的范围（花萼长度）
x1_min = x[:, 0].min()
x1_max = x[:, 0].max()
# 第2维特征的范围（花萼宽度）
x2_min = x[:, 1].min()
x2_max = x[:, 1].max()
# mgrid方法用来生成网格矩阵形式的图框架
x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网络采样点（其实是颜色区域），先沿着x1向右扩展，再沿着x2向下扩展
grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 再通过stack()函数，axis=1，生成测试点，其实就是合并横与纵等于计算x1+x2
grid_value = classifier.predict(grid_test)  # 用训练好的分类器去预测这一片面积内的所有点，为了画出不同类别区域
grid_value = grid_value.reshape(x1.shape)  # （大坑）使刚刚构建的区域与输入的形状相同（裁减掉过多的冗余点，必须写不然会导致越界读取报错，这个点的bug非常难debug）
# 设置两组颜色（高亮色为预测区域，样本点为深色）
light_camp = matplotlib.colors.ListedColormap(['#FFA0A0', '#A0FFA0', '#A0A0FF'])
dark_camp = matplotlib.colors.ListedColormap(['r', 'g', 'b'])
fig = plt.figure(figsize=(10, 5))  # 设置窗体大小
fig.canvas.set_window_title('SVM -2 feature classification of Iris')  # 设置窗体title
# 使用pcolormesh()将预测值（区域）显示出来
plt.pcolormesh(x1, x2, grid_value, cmap=light_camp)
plt.scatter(x[:, 0], x[:, 1], c=y[:, 0], s=30, cmap=dark_camp)  # 加入所有样本点，以深色显示
plt.scatter(test_data[:, 0], test_data[:, 1], c=test_label[:, 0], s=30, edgecolors='white', zorder=2, cmap=dark_camp)
# 单独再把测试集样本点加一个圈,更加直观的查看命中效果
# 设置图表的标题以及x1,x2坐标轴含义
plt.title('SVM -2 feature classification of Iris')
plt.xlabel('length of calyx')
plt.ylabel('width of calyx')
# 设置坐标轴的边界
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.show()