import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from deepsklearn.initializations import init_embedding
from deepsklearn.utils import set_seed
set_seed(42)
'''
std=1.0
OrderedDict([('weight', tensor([[-0.3267, -0.2788, -0.4220],
        [-1.3323, -0.3639,  0.1513],
        [-0.3514, -0.7906, -0.0915],
        [ 0.2352,  2.2440,  0.5817],
        [ 0.4528,  0.6410,  0.5200]]))])
std=0.01
OrderedDict([('weight', tensor([[ 0.0056,  0.0007,  0.0071],
        [-0.0057,  0.0126, -0.0159],
        [-0.0112,  0.0084,  0.0017],
        [-0.0213,  0.0096,  0.0076],
        [ 0.0073, -0.0067,  0.0274]]))])

'''

# =========================
# Model
# =========================
class DNNTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer=nn.Linear(5,5)
        self.embedding=nn.Embedding(5,3)
        self.apply(lambda m:init_embedding(m,std=0.01))

    def forward(self, x):
        return F.relu(self.linear_layer)

model=DNNTiny()

print(model.embedding.state_dict())