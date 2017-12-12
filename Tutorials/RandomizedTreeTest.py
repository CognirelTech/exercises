import time 
import numpy as np

# import pyximport
# pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)

import sys

from numpy import random
import matplotlib.pyplot as plt
from RandomForestsClassifier import RandomForestsClassifier

def main():
    N = 50
    t = np.linspace(1, 2*np.pi, N)
    x1 = random.normal(t*np.cos(t), 0.2)
    y1 = random.normal(t*np.sin(t), 0.2)

    t = np.linspace(1, 2*np.pi, N)
    x2 = random.normal(t*np.cos(t+1.5708), 0.2)
    y2 = random.normal(t*np.sin(t+1.5708), 0.2)

    t = np.linspace(1, 2*np.pi, N)
    x3 = random.normal(t*np.cos(t+1.5708*2), 0.2)
    y3 = random.normal(t*np.sin(t+1.5708*2), 0.2)

    t = np.linspace(1, 2*np.pi, N)
    x4 = random.normal(t*np.cos(t+1.5708*3), 0.2)
    y4 = random.normal(t*np.sin(t+1.5708*3), 0.2)

    data1 = np.vstack((x1, y1)).T
    data2 = np.vstack((x2, y2)).T
    data3 = np.vstack((x3, y3)).T
    data4 = np.vstack((x4, y4)).T

    data = np.vstack((data1, data2))
    data = np.vstack((data, data3))
    data = np.vstack((data, data4))
    print data.shape

    m = np.mean(data, 0)
    var = np.var(data, 0)
    s = np.sqrt(var) + 1e-10

    data = (data - m)/s
    labels = np.ones((N, 1),dtype=np.int32)
    labels = np.vstack((labels, np.ones((N, 1),dtype=np.int32)*2))
    labels = np.vstack((labels, np.ones((N, 1),dtype=np.int32)*3))
    labels = np.vstack((labels, np.ones((N, 1),dtype=np.int32)*4))

    plt.subplot(121)
    colors = ['red']*N + ['yellow']*N + ['green']*N +['blue']*N
    plt.scatter(data[:, 0], data[:, 1], c=colors)
    plt.xlim(-3,3)
    plt.ylim(-3,3)

    numClasses = len(np.unique(labels))


    rForest = RandomForestsClassifier(numTrees=2, stopCriteria='maxDepth', stopValue=9,subspaceSize=500)
    rForest.train(data, labels, data, labels)

    w = 201
    x = np.linspace(-8, 8, 201)
    y = np.linspace(-8, 8, 201)
    x, y = np.meshgrid(x, y)

    x = x.flatten()
    y = y.flatten()
    testData = np.vstack((x, y)).T
    testData = (testData - m)/s
    pred, scores = rForest.predict(testData);

    colors = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]]).T
    colors = colors[:, 0:numClasses]
    rgb = np.dot(colors, scores.T)
    rgb = rgb - np.min(rgb)
    rgb = rgb/np.max(rgb)

    plt.subplot(122)
    plt.axis('equal')
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.imshow(np.reshape(rgb.T, (w, w, 3)), extent=[-3, 3, -3, 3], aspect='auto')

    plt.plot(data[np.nonzero(labels == 1)[0], 0], data[np.nonzero(labels == 1)[0], 1], 
        'o', markerfacecolor='red', markeredgecolor='black')
    plt.plot(data[np.nonzero(labels == 2)[0], 0], data[np.nonzero(labels == 2)[0], 1], 
        'o', markerfacecolor='yellow', markeredgecolor='black')
    plt.plot(data[np.nonzero(labels == 3)[0], 0], data[np.nonzero(labels == 3)[0], 1], 
        'o', markerfacecolor='green', markeredgecolor='black')
    plt.plot(data[np.nonzero(labels == 4)[0], 0], data[np.nonzero(labels == 4)[0], 1], 
        'o', markerfacecolor='blue', markeredgecolor='black')
    plt.show()

if __name__ == "__main__":
    main()
