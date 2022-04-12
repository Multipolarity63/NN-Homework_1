from mnist import *
from model import *
import pandas as pd
import numpy as np
import json

class NN():

    def __init__(self, dim=300, regularization=0.001):
        self.w1 = np.array(pd.read_json("_w1.json"))
        self.w2 = np.array(pd.read_json("_w2.json"))
        self.b1 = np.array(pd.read_json("_b1.json"))
        self.b2 = np.array(pd.read_json("_b2.json"))
        self.regularization = regularization

    def predict(self, data): #输入一个784行n列的矩阵（待预测的数据）
        nn = Model()
        nn.w1 = self.w1
        nn.w2 = self.w2
        nn.b1 = self.b1
        nn.b2 = self.b2
        z1, a1, z2, a2 = nn.forward(data)
        p = np.argmax(a2, axis=0)
        return p
