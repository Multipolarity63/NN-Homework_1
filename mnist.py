import numpy as np
import os
import struct

class DataLoader():
    def __init__(self):
        pass

    # 读取数据
    def mnist(self, path, is_train = True):
        if is_train == 1:
            data = os.path.join(path, "train-images-idx3-ubyte")
            label = os.path.join(path, "train-labels-idx1-ubyte")
        else:
            data = os.path.join(path, "test-images-idx3-ubyte")
            label = os.path.join(path, "test-labels-idx1-ubyte")
        with open(label, "rb") as label_path:
            magic, n = struct.unpack('>II', label_path.read(8))
            labels = np.fromfile(label_path, dtype=np.uint8)
        with open(data, "rb") as data_path:
            magic, num, rows, cols = struct.unpack('>IIII', data_path.read(16))
            images = np.fromfile(data_path, dtype=np.uint8).reshape(len(labels), 784)
        labels = np.eye(10)[labels.reshape(-1)]
        return images, labels





