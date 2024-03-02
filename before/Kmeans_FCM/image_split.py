from scipy import misc
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

face = misc.face()
print(face.shape)
plt.subplot()
plt.imshow(face)
plt.show()



# 分别展示三个通道下的图片
print(face[:,:,0])
print(face[:,:,1])
print(face[:,:,2])
plt.figure(1)
plt.imshow(face[:,:,0])
plt.figure(2)
plt.imshow(face[:,:,1])
plt.figure(3)
plt.imshow(face[:,:,2])
plt.show()


a=face[:,:,1]
image=a.ravel()
print(image.shape)
image.resize(len(image),1)
print(image.shape)
kmeans = KMeans(n_clusters=2)
kmeans.fit(image)#进行聚类
pre = kmeans.predict(image)
pre.resize(768,1024)

plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(a)
plt.subplot(1,2,2)
plt.imshow(pre)
plt.show()

plt.figure(2)
plt.subplot(1,2,1)
plt.imshow(a,cmap=plt.cm.gray)
plt.subplot(1,2,2)
plt.imshow(pre,cmap=plt.cm.gray)
kmeans = KMeans(n_clusters=2)
kmeans.fit(image)#进行聚类
pre = kmeans.predict(image)
pre.resize(768,1024)

plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(a)
plt.subplot(1,2,2)
plt.imshow(pre)

plt.figure(2)
plt.subplot(1,2,1)
plt.imshow(a,cmap=plt.cm.gray)
plt.subplot(1,2,2)
plt.imshow(pre,cmap=plt.cm.gray)

kmeans = KMeans(n_clusters=5)
kmeans.fit(image)#进行聚类
pre = kmeans.predict(image)
pre.resize(768,1024)

plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(a)
plt.subplot(1,2,2)
plt.imshow(pre)

plt.figure(2)
plt.subplot(1,2,1)
plt.imshow(a,cmap=plt.cm.gray)
plt.subplot(1,2,2)
plt.imshow(pre,cmap=plt.cm.gray)


plt.show()