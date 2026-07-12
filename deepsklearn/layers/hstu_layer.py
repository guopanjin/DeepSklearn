import torch
import torch.nn as nn
import torch.nn.functional as F
'''
silu=x*sigmod(x): [-0.28,+infinity]

input =[batch_size,seq_len,embed_dim]
rab:relative attention bias(position encode).
rab_position_table=nn.Embedding(seq_len,num_heads)
rab_time_table=nn.Embedding(bucket_size,num_heads)# split time_diff input bucket_size


q=silu(q_w(input)) #[batch_size,seq_len,embed_dim]
k=silu(k_w(input))
v=silu(v_w(input))
u=silu(u_w(input))

attention_score= silu(q@k/sqrt(head_dim) + rab)/l

attention_score=mask_filled(attention_score)

v=(norm(attention_score@v))*u #u is gate network
hstu_out=out_w(v)
 
out=hstu_out +input
'''
'''

'''
def silu(x):
    return x*F.sigmoid(x)

class HSTULayer(nn.Module):
    def __init__(self,
                 embed_dim:int,
                 num_heads:int,
                 max_len:int
                 ):
        super().__init__()
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.max_len=max_len
        if self.num_heads>self.embed_dim:
            raise ValueError("num_heads must be less than embed_dim")
        if self.embed_dim%self.num_heads!=0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.head_dim=self.embed_dim//self.num_heads
        self.q_w=nn.Linear(self.embed_dim,self.embed_dim)
        self.k_w=nn.Linear(self.embed_dim,self.embed_dim)
        self.v_w=nn.Linear(self.embed_dim,self.embed_dim)
        self.u_w=nn.Linear(self.embed_dim,self.embed_dim)
        self.scale=self.head_dim**0.5
        self.rab_table=nn.Embedding(self.max_len,self.num_heads)
        self.norm=nn.LayerNorm(self.head_dim)
        self.hstu_out_layer=nn.Linear(self.embed_dim,self.embed_dim)
    '''
    x=(batch_size,seq_len,embed_dim)
    padding_mask=(batch_size,seq_len)
    '''
    def forward(self,x,padding_mask=None):
        q=silu(self.q_w(x)) #(batch_size,seq_len,embed_dim)
        k=silu(self.k_w(x))
        v=silu(self.v_w(x))
        u=silu(self.u_w(x))
        batch_size,seq_len,_=q.shape
        valid_seq_len= torch.reshape(torch.sum(padding_mask, dim=1, keepdim=True),(batch_size,1,1,1)) #(batch_size,1,1,1)
        q=torch.reshape(q,(batch_size,seq_len,self.num_heads,self.head_dim)) #(batch_size,seq_len,num_heads,head_dim)
        k=torch.reshape(k,(batch_size,seq_len,self.num_heads,self.head_dim))
        v=torch.reshape(v,(batch_size,seq_len,self.num_heads,self.head_dim))
        u=torch.reshape(u,(batch_size,seq_len,self.num_heads,self.head_dim))

        q=torch.transpose(q,1,2) #(batch_size,num_heads,seq_len,head_dim)
        k = torch.transpose(k, 1, 2)#(batch_size,num_heads,seq_len,head_dim)
        v = torch.transpose(v, 1, 2)#(batch_size,num_heads,seq_len,head_dim)
        u=torch.transpose(u, 1, 2)#(batch_size,num_heads,seq_len,head_dim)
        k=torch.transpose(k,2,3)#(batch_size,num_heads,head_dim,seq_len)
        attention_score=q@k/self.scale#(batch_size,num_heads,seq_len,seq_len)
        #build rab
        row = torch.unsqueeze(torch.arange(self.max_len), 1).expand((self.max_len, self.max_len)).to(q.device)
        col = torch.transpose(row,0,1)
        rab_index=(row - col)
        rab_index[rab_index<0]=0 #[seq_len,seq_len]
        rab=torch.permute(self.rab_table(rab_index),(2,0,1))#[seq_len,seq_len,num_heads]
        #
        attention_weights=silu((attention_score+rab)) #(batch_size,num_heads,seq_len,seq_len)
        #mitigate the impact of weights of the seq_len
        attention_weights=attention_weights/ valid_seq_len
        #padding_mask
        mask = (padding_mask == 0)  # (batch_size,seq_len)
        mask=torch.unsqueeze(mask,dim=1).expand(batch_size,self.max_len,self.max_len)
        mask=torch.unsqueeze(mask,dim=1).expand(batch_size,self.num_heads,self.max_len,self.max_len)
        attention_weights=torch.masked_fill(attention_weights,mask,float(0.0))

        #causal mask,make sure the token can not see the future
        causal_mask=torch.triu(torch.ones(size=(self.max_len,self.max_len),device=q.device),diagonal=1).to(torch.bool)
        attention_weights=torch.masked_fill(attention_weights,causal_mask,float(0.0))

        attention_out=attention_weights@v #(batch_size,num_heads,seq_len,head_dim)
        attention_out=self.norm(attention_out) * u
        attention_out=torch.transpose(attention_out,1,2)#(batch_size,seq_len,num_heads,head_dim)
        hstu_out=torch.reshape(attention_out,(batch_size,seq_len,self.embed_dim))

        hstu_out=self.hstu_out_layer(hstu_out)

        out=hstu_out + x ##(batch_size,seq_len,embed_dim)
        return out









