import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab,n_embd)
        # A learnable weight matrix encodes the position information for each token
        self.position_embedding = nn.Parameter(torch.zeros(n_token,n_embd))
        
    def forward(self,tokens):
        #We apply the embedding first-> (Batch_Size,Seq_Len)->(Batch_Size,Seq_Len,Dim)
        x = self.token_embedding(tokens)
        
        #We add the positional embedding 
        x += self.position_embedding(tokens)
        return x
    
class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        #Pre_attetion norm
        self.prelayer = nn.LayerNorm(n_embd)
        #Self attetion 
        self.attetion = SelfAttention(n_head,n_embd)
        #Pre FNN norm
        self.fnnlayer = nn.LayerNorm(n_embd)
        #2 Feedforward methods for linerization
        self.linear1 = nn.Linear(n_embd,4*n_embd)
        self.linear2 = nn.Linear(4*n_embd,n_embd)
        
    def forward(self,x:torch.Tensor)->torch.Tensor:
        #x:(Batch_Size,Seq_Len,Dim)
        #Firstly a residual layer
        residue = x
        
        #Than we apply layer normalization into self attention 
        x = self.prelayer(x)
        x = self.attetion(x,casual_mask=True)#Causal mask to prevent the influence of earlier tokens to future token
        
        #Residual Connection
        x+=residue
        
        #Feed Forward Layer
        # Apply a feedforward layer where the hidden dimension is 4 times the embedding dimension. 
        residue = x
        
        #Normailzation and feed forward process
        x=self.fnnlayer(x)
        x=self.linear1(x)
        
        #Use Quick GelU activation function
        x = x*torch.sigmoid(1.702*x)
        x = self.linear_2(x)
        
        #Residual Connection
        x += residue
        return x

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)
        
        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])
        
        self.layernorm = nn.LayerNorm(768)
        
    def forward(self,x:torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        state = self.embedding(tokens)        
        # Apply encoder layers similar to the Transformer's encoder.
        for layer in self.layers: 
            # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
            state = layer(state)
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.layernorm(state)
        
        return output