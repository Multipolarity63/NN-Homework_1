import numpy as np
from mnist import *
from model import *
import pandas as pd
from tqdm import tqdm
import json


#构建迭代器生成batch
def data_iter(data, label, size):
    np.random.seed(1000)
    n = data.shape[0]
    iters = list(np.random.permutation(n))
    for i in range(n // size):
        data_batch = data[iters[(i * size):((i + 1) * size)], :]
        label_batch = label[iters[(i * size):((i + 1) * size)], :]
        yield data_batch, label_batch

num_epochs = 1000 # 训练的epoch数量
size = 1024 # 每个batch的大小
learning_rate = 0.000001 # 学习率
decay_rate = 0.9
w1_path = [] # 保存参数中间过程的列表
w2_path = []
b1_path = []
b2_path = []
train_loss = [] # 保存训练集loss的列表
test_acc = []
test_loss = []

dataloader = DataLoader()
(train_data, train_label) = dataloader.mnist("./data")
(test_data, test_label) = dataloader.mnist("./data", is_train=False)

model = Model(dim=300, regularization=0.001)
for epoch in tqdm(range(num_epochs)):
    w1 = model.w1
    w2 = model.w2
    b1 = model.b1
    b2 = model.b2
    w1_path.append(np.sum(np.square(w1)))  # 记录参数F范数的平方
    w2_path.append(np.sum(np.square(w2)))
    b1_path.append(np.sum(np.square(b1)))
    b2_path.append(np.sum(np.square(b2)))
    (z1, a1, z2, a2) = model.forward(np.transpose(train_data))
    train_loss.append(model.loss(a2, np.transpose(train_label)))
    (z1, a1, z2, a2) = model.forward(np.transpose(test_data))
    test_loss.append(model.loss(a2, np.transpose(test_label)))
    prediction = model.pred(a2)
    true = model.pred(np.transpose(test_label))
    test_acc.append(model.accuracy(prediction, true))
    if epoch // 25 > (epoch-1) // 25:
        learning_rate = learning_rate*decay_rate
    for x, y in data_iter(train_data, train_label, size):
        y = np.transpose(y)
        x = np.transpose(x)
        (z1, a1, z2, a2) = model.forward(x)
        (dw2, db2, dw1, db1) = model.backward(x, y, z1, a1, z2, a2)
        model.SGD(dw2, db2, dw1, db1, 0.95, learning_rate)

df = pd.DataFrame(w1)
df.to_json("_w1.json")
df = pd.DataFrame(w2)
df.to_json("_w2.json")
df = pd.DataFrame(b1)
df.to_json("_b1.json")
df = pd.DataFrame(b2)
df.to_json("_b2.json")
df = pd.DataFrame(train_loss)
df.to_json("_train_loss.json")
df = pd.DataFrame(test_loss)
df.to_json("_test_loss.json")
df = pd.DataFrame(test_acc)
df.to_json("_test_acc.json")
df = pd.DataFrame(w1_path)
df.to_json("_w1_path.json")
df = pd.DataFrame(w2_path)
df.to_json("_w2_path.json")
df = pd.DataFrame(b1_path)
df.to_json("_b1_path.json")
df = pd.DataFrame(b2_path)
df.to_json("_b2_path.json")

Otherpara = {"dim1":300, "regularization":0.001, "learning_rate":0.000001,"momentum":0.95,"lr_decay":0.8,"decay_step":25}
js = json.dumps(Otherpara)
fp = open('Otherpara.json', 'a')
fp.write(js)
fp.close()



