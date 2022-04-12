import numpy as np


class Model():

    # 定义属性参数和正则化系数
    def __init__(self, dim=300, regularization=0.001):
        self.w1 = np.random.normal(loc = 0.0, scale = 0.05, size = (dim,784))
        self.w2 = np.random.normal(loc = 0.0, scale = 0.05, size = (10, dim))
        self.b1 = np.zeros((dim,1))
        self.b2 = np.zeros((10,1))
        self.regularization = regularization

    # 定义两种激活函数
    def relu(self,x):
        y = np.maximum(0,x)
        return y

    def softmax(self,x):
        y = np.exp(x) / (np.sum(np.exp(x), axis=0, keepdims=True))
        return y

    # 定义损失函数
    def loss(self,x,y):
        n = x.shape[1]
        regularization = self.regularization
        w1 = self.w1
        w2 = self.w2
        loss = -np.sum(np.multiply(y,np.log(x))) + regularization*(np.sum(np.square(w1)) + np.sum(np.square(w2)))
        return loss

    # 前向传播
    def forward(self, data):
        w1 = self.w1
        w2 = self.w2
        b1 = self.b1
        b2 = self.b2
        z1 = np.dot(w1,data) + b1
        a1 = self.relu(z1)
        z2 = np.dot(w2,a1) + b2
        a2 = self.softmax(z2)
        return z1,a1,z2,a2

    # 反向传播得梯度
    def backward(self, data, label, z1, a1, z2, a2):
        regularization = self.regularization
        n = data.shape[1]
        w1 = self.w1
        w2 = self.w2
        b1 = self.b1
        b2 = self.b2
        dz2 = a2 - label
        dw2 = np.dot(dz2, np.transpose(a1)) + 2*regularization*w2
        db2 = np.sum(dz2, axis=1, keepdims=True)
        da1 = np.dot(np.transpose(w2), dz2)
        dz1 = np.multiply(da1, a1>0)
        dw1 = np.dot(dz1, np.transpose(data)) + 2*regularization*w1
        db1 = np.sum(dz1, axis=1, keepdims=True)
        return dw2, db2, dw1, db1

    # SGD求优化
    def SGD(self, dw2, db2, dw1, db1, mu, step):
        v_w2 = 0
        v_b2 = 0
        v_w1 = 0
        v_b1 = 0
        v_w2 = mu*v_w2 - step*dw2
        self.w2 += v_w2
        v_b2 = mu * v_b2 - step * db2
        self.b2 += v_b2
        v_w1 = mu * v_w1 - step * dw1
        self.w1 += v_w1
        v_b1 = mu * v_b1 - step * db1
        self.b1 += v_b1

    # 单次预测结果
    def pred(self, a2):
        p = np.argmax(a2, axis=0)
        return p

    # 计算准确率
    def accuracy(self, x ,y):
        n = len(x)
        rate = sum(x==y)/n
        return rate







