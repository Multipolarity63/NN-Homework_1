from mnist import *
from model import *
from NN import *
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

n=1000
e_model = NN()
dataloader = DataLoader()
(test_data, test_label) = dataloader.mnist("./data", is_train=False)
example_data = test_data[n,:]
example_data = example_data.reshape(784,1)
pred = e_model.predict(example_data)
print(pred)
print(np.argmax(test_label[n],axis=0))



