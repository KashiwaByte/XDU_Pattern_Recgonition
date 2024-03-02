from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import math

#prepare the data
iris = datasets.load_iris()
X = iris.data
y = iris.target
names = iris.feature_names
labels = iris.target_names
y_c = np.unique(y)

"""visualize the distributions of the four different features in 1-dimensional histograms"""
fig, axes = plt.subplots(2, 2, figsize=(12, 6))
for ax, column in zip(axes.ravel(), range(X.shape[1])):
  # set bin sizes
  min_b = math.floor(np.min(X[:, column]))
  max_b = math.ceil(np.max(X[:, column]))
  bins = np.linspace(min_b, max_b, 25)

  # plotting the histograms
  for i, color in zip(y_c, ('blue', 'red', 'green')):
      ax.hist(X[y == i, column], color=color, label='class %s' % labels[i],
              bins=bins, alpha=0.5, )
  ylims = ax.get_ylim()

  # plot annotation
  l = ax.legend(loc='upper right', fancybox=True, fontsize=8)
  l.get_frame().set_alpha(0.5)
  ax.set_ylim([0, max(ylims) + 2])
  ax.set_xlabel(names[column])
  ax.set_title('Iris histogram feature %s' % str(column + 1))

  # hide axis ticks
  ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                 labelbottom=True, labelleft=True)

  # remove axis spines
  ax.spines['top'].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.spines["bottom"].set_visible(False)
  ax.spines["left"].set_visible(False)

axes[0][0].set_ylabel('count')
axes[1][0].set_ylabel('count')
fig.tight_layout()



np.set_printoptions(precision=4)


mean_vector = []  # 类别的平均值
for i in y_c:
    mean_vector.append(np.mean(X[y == i], axis=0))
    print('Mean Vector class %s:%s\n' % (i, mean_vector[i]))
S_W = np.zeros((X.shape[1], X.shape[1]))
for i in y_c:
    Xi = X[y == i] - mean_vector[i]
    S_W += np.mat(Xi).T * np.mat(Xi)
print('S_W:\n', S_W)

S_B = np.zeros((X.shape[1], X.shape[1]))
mu = np.mean(X, axis=0)  # 所有样本平均值
for i in y_c:
    Ni = len(X[y == i])
    S_B += Ni * np.mat(mean_vector[i] - mu).T * np.mat(mean_vector[i] - mu)
print('S_B:\n', S_B)
eigvals, eigvecs = np.linalg.eig(np.linalg.inv(S_W) * S_B)  # 求特征值，特征向量
np.testing.assert_array_almost_equal(np.mat(np.linalg.inv(S_W) * S_B) * np.mat(eigvecs[:, 0].reshape(4, 1)),
                                     eigvals[0] * np.mat(eigvecs[:, 0].reshape(4, 1)), decimal=6, err_msg='',
                                     verbose=True)
# sorting the eigenvectors by decreasing eigenvalues
eig_pairs = [(np.abs(eigvals[i]), eigvecs[:, i]) for i in range(len(eigvals))]
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
W = np.hstack((eig_pairs[0][1].reshape(4, 1), eig_pairs[1][1].reshape(4, 1)))
X_trans = X.dot(W)
assert X_trans.shape == (150, 2)
plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.scatter(X_trans[y == 0, 0], X_trans[y == 0, 1], c='r')
plt.scatter(X_trans[y == 1, 0], X_trans[y == 1, 1], c='g')
plt.scatter(X_trans[y == 2, 0], X_trans[y == 2, 1], c='b')
plt.title('Fisher result')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend(labels, loc='best', fancybox=True)
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import math

# prepare the data
iris = datasets.load_iris()
X = iris.data
y = iris.target
names = iris.feature_names
labels = iris.target_names
y_c = np.unique(y)

"""visualize the distributions of the four different features in 1-dimensional histograms"""
fig, axes = plt.subplots(2, 2, figsize=(12, 6))
for ax, column in zip(axes.ravel(), range(X.shape[1])):
    # set bin sizes
    min_b = math.floor(np.min(X[:, column]))
    max_b = math.ceil(np.max(X[:, column]))
    bins = np.linspace(min_b, max_b, 25)

    # plotting the histograms
    for i, color in zip(y_c, ('blue', 'red', 'green')):
        ax.hist(X[y == i, column], color=color, label='class %s' % labels[i],
                bins=bins, alpha=0.5, )
    ylims = ax.get_ylim()

    # plot annotation
    l = ax.legend(loc='upper right', fancybox=True, fontsize=8)
    l.get_frame().set_alpha(0.5)
    ax.set_ylim([0, max(ylims) + 2])
    ax.set_xlabel(names[column])
    ax.set_title('Iris histogram feature %s' % str(column + 1))

    # hide axis ticks
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                   labelbottom=True, labelleft=True)

    # remove axis spines
    ax.spines['top'].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

axes[0][0].set_ylabel('count')
axes[1][0].set_ylabel('count')
fig.tight_layout()
plt.show()