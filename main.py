import numpy as np
import sklearn.svm as svm
import matplotlib as mpl
import matplotlib.pyplot as plt

def extend(a, b, r=0.02):
    return a * (1 + r) - b * r, -a * r + b * (1 + r)

def startjob():
    #[-5. -3. -1.  1.  3.  5.]
    t = np.linspace(-5, 5, 6)
    print(t)
    # [X, Y] = meshgrid(x, y)
    # 将向量x和y定义的区域转换成矩阵X和Y, 其中矩阵X的行向量是向量x的简单复制，
    # 而矩阵Y的列向量是向量y的简单复制(注：下面代码中X和Y均是数组，在文中统一称为矩阵了)。
    # 假设x是长度为m的向量，y是长度为n的向量，则最终生成的矩阵X和Y的维度都是
    # nm （注意不是mn）。
    #参见: https://zhuanlan.zhihu.com/p/29663486
    t1, t2 = np.meshgrid(t, t)
    # t1:
    # [[-5. -3. -1.  1.  3.  5.]
    #  [-5. - 3. - 1.  1.  3.  5.]
    #  [-5. - 3. - 1. 1. 3. 5.]
    #  [-5. - 3. - 1. 1. 3. 5.]
    #  [-5. - 3. - 1. 1. 3. 5.]]
    #t2
    #[[-5. -5. -5. -5. -5. -5.]
    #[-3. -3. -3. -3. -3. -3.]
    #[-1. -1. -1. -1. -1. -1.]
    #[ 1.  1.  1.  1.  1.  1.]
    #[ 3.  3.  3.  3.  3.  3.]
    #[ 5.  5.  5.  5.  5.  5.]]
    x1 = np.stack((t1.ravel(), t2.ravel()), axis=1)
    # print(x1)
    N = len(x1)
    x2 = x1 + (1, 1)
    # print(x2)
    x = np.concatenate((x1, x2))
    y = np.array([1] * N + [-1] * N)
    print(x, y)
    clf = svm.SVC(C=0.1, kernel='rbf', gamma=5)
    clf.fit(x, y)
    y_hat = clf.predict(x)
    print('准确率: %.1f%%' % (np.mean(y_hat == y) * 100))

    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FFA0A0'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r'])
    x1_min, x1_max = extend(x[:, 0].min(), x[:, 0].max()) #第0列的范围
    print('x1_min: \t', x1_min)
    print('x1_max: \t', x1_max)
    x2_min, x2_max = extend(x[:, 1].min(), x[:, 1].max()) #第1列的范围
    print('x2_min: \t', x2_min)
    print('x2_max: \t', x2_max)
    x1, x2 = np.mgrid[x1_min:x1_max:300j, x2_min:x2_max:300j] #生成网格采样点
    print('x1: \n', x1)
    print('x2: \n', x2)
    #numpy.ndarray.flat: 将所有数据拉成一行
    print('x1.flat: \n', x1.flat, np.array(x1.flat))
    #<numpy.flatiter object at 0x00000000043B6870>
    #[-5.22 -5.22 -5.22 ...  6.22  6.22  6.22]
    print('x2.flat: \n', x2.flat, np.array(x2.flat))
    #<numpy.flatiter object at 0x00000000043B6870>
    #[-5.22 -5.18173913 -5.14347826 ...  6.14347826  6.18173913 6.22]

    #np.stack, 当axis=1时，新的行向量的值为原有两个array对应列向量索引的值的叠加
    grid_test = np.stack((x1.flat, x2.flat), axis= 1) #测试点
    print('grid_test: \n', grid_test)
    # [[-5.22 -5.22]
    #  [-5.22 -5.18173913]
    #  [-5.22 -5.14347826]
    #      ...
    #  [6.22 6.14347826]
    # [6.22 6.18173913]
    # [6.22 6.22]]

    grid_hat = clf.predict(grid_test)
    grid_hat.shape = x1.shape
    print('grid_hat: \n', grid_hat)

    plt.figure(facecolor='w')
    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
    plt.scatter(x[:,0], x[:,1], s=60, c=y, marker='o', cmap=cm_dark)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title('SVM的RBF核与过拟合', fontsize=18)
    plt.tight_layout(0.2)
    plt.show()



if __name__ == '__main__':
    startjob()