import torch
import torch.nn as nn

class FMInteraction(nn.Module):
    def __init__(self,
                 is_sum=True
                 ):
        super(FMInteraction, self).__init__()
        self.is_sum=is_sum
    # x =(batch_size,feature_size,embedding_dim)
    def forward(self,x):
        first_part=torch.pow(torch.sum(x,dim=1),2)#(batch_size,embedding_dim)
        seconde_part=torch.sum(torch.pow(x,2),dim=1)
        if self.is_sum:
            output=0.5*torch.sum(first_part-seconde_part,dim=1,keepdim=True)#(batch_size,1)
        else:
            output = 0.5 * (first_part - seconde_part)  # (batch_size,embedding_dim)
        return output


