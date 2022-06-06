from fate_torch.nn import Linear, ReLU, Sigmoid, Embedding, Conv1d, BatchNorm1d, Dropout
from fate_torch.init import normal_, xavier_normal_, constant_
from fate_torch.base import Sequential
from fate_torch.serialization import recover_sequential_from_json
import numpy as np
from torch import Tensor


layer_a = Linear(64, 32, True)
# normal_(layer_a)
# normal_(layer_a, mean=0.5, init='bias')

layer_b = Linear(32, 16, True)
# xavier_normal_(layer_b)
# constant_(layer_b, 1, init='bias')

layer_c = Linear(16, 8, True)

seq = Sequential(
    layer_a,
    ReLU(),
    layer_b,
    Dropout(),
    ReLU(),
    layer_c
)

normal_(seq)
seq2 = Sequential()
# print(seq.to_json())
# seq2 = recover_sequential_from_json(seq.to_json())
#
# t1 = Tensor(np.random.random((1, 64)))
