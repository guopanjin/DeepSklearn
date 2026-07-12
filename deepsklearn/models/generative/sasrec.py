import torch
import torch.nn as nn
import torch.nn.functional as F
from deepsklearn.layers import EncoderLayer
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

class SASRec(nn.Module):
    def __init__(self,
                 num_items: int,
                 embed_dim: int = 32,
                 seq_len: int = 50,
                 num_layers: int = 2,
                 num_heads: int = 2
                 ):
        super().__init__()
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.item_embedding_table = nn.Embedding(self.num_items + 1, self.embed_dim, padding_idx=0)
        self.position_embedding_table=nn.Embedding(self.seq_len,self.embed_dim)

        encoder_layer_list = []
        for i in range(self.num_layers):
            encoder_layer = EncoderLayer(embed_dim=self.embed_dim,
                                    num_heads=self.num_heads,
                                    ffn_dim=self.embed_dim*4,
                                    use_causal_mask=True,
                                    norm_first=True
                                   )
            encoder_layer_list.append(encoder_layer)
        self.encoder_layer_module_list = nn.ModuleList(encoder_layer_list)
        self.apply(lambda m: init_embedding(m))
        # clear the gradient for   item_embedding_table.weight,because the nn.Embedding of padding_idx=0
        # will not be effective for it
        self.item_embedding_table.weight.register_hook(mask_grad)

    def forward(self, x):
        item_sequence = x["item_sequence"]
        padding_mask = x["padding_mask"]
        padding_mask=(padding_mask==0)
        x = self.item_embedding_table(item_sequence)  # (batch_size,seq_len,embed_dim)
        position_embedding=self.position_embedding_table(torch.arange(self.seq_len).to(x.device))#(seq_len,embedding_dim)
        x=x+position_embedding
        for encoder_layer in self.encoder_layer_module_list:
            x = encoder_layer(x, key_padding_mask=padding_mask)

        item_embeddings = torch.transpose(self.item_embedding_table.weight, 0, 1)  # (embed_dim,num_items+1)
        logits = x @ item_embeddings  # (batch_size,seq_len,num_items+1)
        return logits


def mask_grad(grad):
    new_grad = grad.clone()
    new_grad[0] = 0
    return new_grad






