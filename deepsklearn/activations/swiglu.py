import torch
import torch.nn as nn
import torch.nn.functional as F
'''
x-->x*v
x--->x*w--->silu(x*w)
x*v*(silu(x*w))
input: (batch_size,dim)
output:(batch_size,dim)
'''
class SwiGlu(nn.Module):
    def __init__(self,dim,hidden):
        super().__init__()
        self.dim=dim
        self.hidden=hidden
        self.v=nn.Linear(self.dim,self.hidden,bias=False)
        self.w=nn.Linear(self.dim,self.hidden,bias=False)
        self.out=nn.Linear(self.hidden,self.dim,bias=False)
    def forward(self,x):
        x=self.v(x)*F.silu(self.w(x)) #(batch_size,hidden_size)
        return self.out(x)#(batch_size,dim)