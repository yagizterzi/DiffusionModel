import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    #d_embed = Number of channels of each pixel 
    
    def __init__(self, n_heads:int, d_embed:int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        #Wq Wk and Wv matrices represented as one big linear layer instead of 3 matrices
        self.in_proj=nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        #Wo matrix
        self.out_proj  =  nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        #Dimension of each head
        self.d_head = d_embed // n_heads
        
    #Causal mask to prevent the influence of earlier tokens to future token
    def forward(self,x:torch.Tensor,causal_mask = False):
        #x: (Batch_Size, Seq_len , dim)
        input_shape=x.shape
        
        batch_size,sequence_len,d_embed = input_shape
        # (Batch_Size, Seq_Len, H, Dim / H), H: amount of heads
        mid_shape = (batch_size, sequence_len, self.n_heads, self.d_head) 
        
        #We apply input projection to the input and turn it into Query,Key,Value matrices by seperating input projection into 3 matrices
        ## (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 3) -> 3 tensor of shape (Batch_Size, Seq_Len, Dim)
        
        q,k,v = self.in_proj(x).chunk(3, dim=-1)
        
        #Split the values according to number of head
        #(Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        q = q.view(mid_shape).transpose(1,2)
        k = k.view(mid_shape).transpose(1,2)
        v = v.view(mid_shape).transpose(1,2)
        
        #Calculate the attention
        
        #(Batch_Size, H, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-1,-2)
        if causal_mask:
             #Mask where the upper triangle (above the principal diagonal) is 1
            mask=torch.ones_like(weight , dtype=torch.bool).triu(1)
            #Fill the upper triangle with -inf
            weight.masked_fill_(mask, -torch.inf) 
        
        #Divide it by sqrroot of the model and apply softmax
        weight/=math.sqrt(self.d_head)
        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = F.softmax(weight,dim=-1)
        
        #apply weights
        # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        output =  weight @ v
        
        # (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, Seq_Len, H, Dim / H)
        output = output.transpose(1,2)
        
        output = output.reshape(input_shape)
        
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output =  self.out_proj(output)
        
        return output
class CrossAttention(nn.Module):
    def __init__(self, n_heads,d_embd,d_cross,in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        #Key Query Value matrices defined indivually instead of one big linear matrix 
        self.query_proj = nn.Linear(d_embd,d_embd,bias=in_proj_bias)
        self.key_proj = nn.Linear(d_cross,d_embd,bias=in_proj_bias)
        self.value_proj = nn.Linear(d_cross, d_embd, bias=in_proj_bias)
        #Wo matrix
        self.output = nn.Linear(d_embd,d_embd,bias=out_proj_bias)
        self.n_heads = n_heads
        #Dimension of each head
        self.d_heads = d_embd//n_heads
    def forward(self,x,y):
        #X:(Latent) = (Batch_Size, Seq_Len_Q, Dim_Q)
        #Y:(Context) = (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768)
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        # Divide each embedding of Q into multiple heads such that d_heads * n_heads = Dim_Q
        mid_shape = (batch_size, -1, self.n_heads, self.d_heads)
        
        #Multiply key query and value with its matrices
        q=self.query_proj(x)
        k=self.key_proj(y)
        v=self.value_proj(y)
        
        #Split them into H number of heads
        q = q.view(mid_shape).transpose(1, 2) #(Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        k = k.view(mid_shape).transpose(1, 2) #(Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        v = v.view(mid_shape).transpose(1, 2) #(Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        
        #Calculating attention
        weight = q @ k.transpose(-1,-2)
        weight/=math.sqrt(self.d_heads)
        #Now apply softmax 
        weight = F.softmax(weight, dim=-1)
        #Note:We will not use a casual mask this time because in this we will match the prompts with pixels unlike self attention 
        #Now we will aply these weights to the value so we can get the output 
        output = weight @ v
        output = output.transpose(1, 2).contiguous()#We need our tensor to be contigious for our operations
        output = output.view(input_shape)#(Batch_Size, Seq_Len_Q, Dim_Q)
        output = self.output(output)
        return output
        
        
        