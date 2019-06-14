import numpy as np
from matplotlib import pyplot as plt
'''
3N in the Input Layer
4N in the Hidden Layer
1N in the Output Layer
'''

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def sigmoid_p(x):
    return sigmoid(x) * (1-sigmoid(x))

def loss(a, y):
    if y == 1:
        return -np.log(a)
    else:
        return -np.log(1-a)

def loss_d(a, y):
    if y == 1:
        return -1/a
    else:
        return 1/(1-a)

InputData = np.array([[0, 0, 0],
               [0, 0, 1],
               [0, 1, 0],
               [1, 0, 0],
               [1, 0, 1],
               [1, 1, 1]])

TargetData = np.array([[0], [1], [1], [0], [1], [0]])

TestData = np.array([[1, 1, 0],
                     [0, 1, 1]])

w1 = np.zeros((4, 3))
b1 = np.random.randn(1, 4)
b1 = np.repeat(b1, 6, axis=0)
print(b1)

w2 = np.zeros((1, 4))
b2 = np.zeros((1,1))

iterations = 1
lr = 0.1
costlist = []

for i in range(iterations):

    z1 = np.dot(w1, InputData.T) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(w2, z1)

    print(z1)
