import torch
import torch.nn as nn
import torch.nn.functional as F
from deepsklearn.initializations import init_embedding
'''
paper hyperParameter:

1: input:
  x=(batch_size,seq_len)
  padding_mask=(batch_size,seq_len)
2: 
  item_embedding #(num_item+1,embed_dim)
  x=self.item_embedding(x)#(batch_size,seq_len,embed_dim)
  x=x+position_encoding
  x=self.encoder_layer(x,padding_mask=padding_mask)# 2 layers  (batch_size,seq_len,embed_dim)

  logits=x@self.item_embedding^T #(batch_size,seq_len,num_items+1)
'''

class GRU4Rec(nn.Module):
    def __init__(self,
                 num_items: int,
                 embed_dim: int = 32,
                 seq_len: int = 50,
                 num_gru_layers: int = 2
                 ):
        super().__init__()
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.num_gru_layers=num_gru_layers
        self.item_embedding_table = nn.Embedding(self.num_items + 1, self.embed_dim, padding_idx=0)
        self.gru=nn.GRU(input_size=self.embed_dim, num_layers=self.num_gru_layers, hidden_size=self.embed_dim,batch_first=True)
        self.apply(lambda m: init_embedding(m))
        # clear the gradient for   item_embedding_table.weight,because the nn.Embedding of padding_idx=0
        # will not be effective for it
        self.item_embedding_table.weight.register_hook(mask_grad)

    def forward(self, x):
        item_sequence = x["item_sequence"]
        x = self.item_embedding_table(item_sequence)  # (batch_size,seq_len,embed_dim)
        x,_=self.gru(x)
        item_embeddings = torch.transpose(self.item_embedding_table.weight, 0, 1)  # (embed_dim,num_items+1)
        logits = x @ item_embeddings  # (batch_size,seq_len,num_items+1)
        return logits


def mask_grad(grad):
    new_grad = grad.clone()
    new_grad[0] = 0
    return new_grad