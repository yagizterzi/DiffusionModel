import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, n_embd:int):
        super().__init__()
        self.linear1 = nn.Linear(n_embd,4*n_embd)
        self.linear2 = nn.Linear(4*n_embd,4*n_embd)
        
    def forward(self,x:torch.Tensor)->torch.Tensor:
        # x:(1,320)
        x=self.linear1(x)
        
        x=F.silu(x)
        
        x=self.linear2(x)
        
        #x:(1,1280)
        return x

class SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                # Attention layers need both x and context
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                # Residual blocks need both x and time embedding
                x = layer(x, time)
            else:
                # Standard layers just need x
                x = layer(x)
        return x
class UpSample(nn.Module):
    def __init__(self, channels:int):
        super().__init__()
        self.conv=nn.Conv2d(channels,channels,kernel_size=3,padding=1)
    def forward(self,x):
        #Height and Width *2 others stay same
        x = F.interpolate(x,scale_factor=2,mode="nearest")#Same upsampling as in autoencoder
        return self.conv(x)
    
class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels,out_channels,n_time=1280):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32,in_channels)
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.linear_time = nn.Linear(n_time,out_channels)
        
        self.groupnorm2 = nn.GroupNorm(32,out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()#Direct connection 
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)#Convert the size of input with convolution so we can connect them 
    def forward(self,feature,time):
        #Feature(latent):(Batch_size,in_channels,Height,Width)
        #Time:(1,1280)
        
        residue = feature
        
        feature = self.groupnorm(feature)
        
        feature =F.silu(feature)
        
        #Feature:(Batch_Size, Out_Channels, Height, Width)
        feature =self.conv(feature)
        
        time = F.silu(time)
        
        #Time :(1,Out_Channels)
        time =self.linear_time(time)
        #Unsqueeze makes time 4 dimensional like feature 
        # (Batch_Size, Out_Channels, Height, Width) + (1, Out_Channels, 1, 1) -> (Batch_Size, Out_Channels, Height, Width
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        
        merged = self.groupnorm2(merged)
        
        merged = F.silu(merged)
        
        merged = self.conv2(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) + (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        return merged + self.residual_layer(residue)
        
        
class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head:int,n_embd:int , d_context = 768):
        super().__init__()
        channels = n_embd*n_head   
        
        # Group normalization layer
        self.groupnorm = nn.GroupNorm(32,channels,eps=1e-6)
        # Convolutional layer for input transformation
        self.conv = nn.Conv2d(channels,channels,kernel_size=1,padding=0)
        
        # Layer normalization before self-attention
        self.Layernorm1 = nn.LayerNorm(channels)
        # Self-attention mechanism
        self.selfattention = SelfAttention(n_head,channels,in_proj_bias=False)
        # Layer normalization before cross-attention
        self.Layernorm2 = nn.LayerNorm(channels)
        # Cross-attention mechanism (attends to context)
        self.crossattention = CrossAttention(n_head,channels,d_context,in_proj_bias=False)
        # Layer normalization after attention blocks
        self.Layernorm3 = nn.LayerNorm(channels)
        
        # First linear layer for the GEGLU (Gated Linear Unit) activation
        self.Lineargeglu1 = nn.Linear(channels,4*channels*2)
        # Second linear layer for the GEGLU activation, reducing dimensionality
        self.Lineargeglu2 = nn.Linear(4*channels,channels)
        
        # Convolutional layer for output transformation
        self.conv_out = nn.Conv2d(channels,channels,kernel_size=1,padding=0)
    def forward(self,x,context):
        #x : (Batch_Size, Features , Height , Width)(Latent)
        #context : (Batch_Size , Seq_Len , Dim(d_context))(Prompt)
        
        #Firstly we have to apply normalization and convolution to our latent
        #Resuidual Connection layer 
        residue = x
        
        x=self.groupnorm(x)
        
        x=self.conv(x)
        
        #Then we get the shape and make it fit for attention(get height*width,transpose it)
        n,c,h,w = x.shape
        
        x=x.view((n,c,h*w))
        
        #(Batch_Size, Height * Width, Features)
        x=x.transpose(-1,-2)
        
        #Now we will do Normalization + Self Attention with skip connection
        small_residue = x 
        
        x = self.Layernorm1(x)
        
        x = self.selfattention(x)
        
         # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += small_residue
        
        #Now we will apply Normalization + Cross Attention with skip connection
        small_residue = x
        
        x = self.Layernorm2
        
        x = self.crossattention(x,context)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += small_residue
        
        #Now we will apply the last normalization and after that we will apply FF layer with GEGLU and skip connection
        x = small_residue
        
        x = self.Layernorm3(x)
        
        #GEGLU(x) = GELU(Linear_Projection_Part1(x)) * Linear_Projection_Part2(x)
        #(Batch_Size, Height * Width, Features) -> two tensors of shape (Batch_Size, Height * Width, Features * 4)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1) 
        
        # Element-wise product: (Batch_Size, Height * Width, Features * 4) * (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features * 4)
        x = x * F.gelu(gate)
        
         # (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features)
        x = self.Lineargeglu2(x)
        
        x += small_residue
        #Now we will change our tensor to not be sequence of pixels 
        x = x.transpose(-1,-2)
        
        x = x.view(n,c,h,w)
        
        # Final skip connection between initial input and output of the block
        # (Batch_Size, Features, Height, Width) + (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        return self.conv_out(x) + residue
        
        
        
        
        
    
class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        # U-Net consists of three main parts: 
        # an encoder that downsamples the input through convolutional layers, 
        # a bottleneck that processes the most compact representation, and 
        # a decoder that upsamples with skip connections from the encoder to restore spatial details
        
        # First Encoders:
        self.encoders = nn.ModuleList([
           # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
           SwitchSequential(nn.Conv2d(4,320,kernel_size=3,padding=1)),
           
           # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
           SwitchSequential(UNET_ResidualBlock(320,320),UNET_AttentionBlock(8,40)),
           
           # (Batch_Size, 320, Height / 8, Width / 8) ->  (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
           SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
           
           # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 16, Width / 16)
           SwitchSequential(nn.Conv2d(320,320,kernel_size=3,stride=2,padding=1)),
           
           # (Batch_Size, 320, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
           SwitchSequential(UNET_ResidualBlock(320,640),UNET_AttentionBlock(8,80)),
           
           # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
           SwitchSequential(UNET_ResidualBlock(640, 640),UNET_AttentionBlock(8,80)),
           
           # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 32, Width / 32)
           SwitchSequential(nn.Conv2d(640,640,kernel_size=3,stride=2,padding=1)),
           
           # (Batch_Size, 640, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
           SwitchSequential(UNET_ResidualBlock(640, 1280),UNET_AttentionBlock(8,160)),
           
           # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
           SwitchSequential(UNET_ResidualBlock(1280, 1280),UNET_AttentionBlock(8,160)),
           
           # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280 , Height/ 64, Width/ 64)
           SwitchSequential(nn.Conv2d(1280,1280,kernel_size=3,stride=2,padding=1)),
           
           # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
           SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            
           # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
           SwitchSequential(UNET_ResidualBlock(1280, 1280)),  
        ])
        
        # Second: The BottleNeck
        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280,1280),
            UNET_AttentionBlock(8,160),
            UNET_ResidualBlock(1280,1280)
            # Size remains (Batch_Size, 1280, Height / 64, Width / 64) on entire bottleneck
        )
        
        # Lastly: The Decoder
        # Decoder is basically inverse of the encoder(num_features ↓ , image size ↑ )
        self.decoders = nn.ModuleList([
           # 2560 Features due to skip connection between encoders last layer and decoders first layer doubles the feature size 
           # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
           SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
           # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
           SwitchSequential(UNET_ResidualBlock(2560, 1280)),
           
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 32, Width / 32) 
           SwitchSequential(UNET_ResidualBlock(2560, 1280),UpSample(1280)),
           
           # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
           SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
           
           # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
           SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
           
           # (Batch_Size, 1920, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 16, Width / 16)
           # +1280 due to connection 
           SwitchSequential(UNET_ResidualBlock(1920,1280),UNET_AttentionBlock(8,160),UpSample(1280)),
           
           #(Batch_Size, 1920, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
           SwitchSequential(SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80))),
           
           # (Batch_Size, 1920, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
           SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 1280, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
           SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 960, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 8, Width / 8)
           SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), UpSample(640)),
            
            # (Batch_Size, 960, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
           SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
           SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
           SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
           
        ])
class UNET_Output(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32,in_channels)
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
    def forward(self,x):
       #x: (Batch_Size, 320, Height / 8, Width / 8)
       x= self.groupnorm(x)
       
       x=F.silu(x)
       
       x=self.conv(x)
       #x: (Batch_Size, 4, Height / 8, Width / 8)
       return x
        
class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        # We will convert the timestamp which when image is noisified into an embedding
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.unet_output = UNET_Output(320,4)
    def forward(self,latent:torch.Tensor,context:torch.Tensor,time:torch.Tensor):
        # latent: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim)
        # time: (1, 320)
        
        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)
        
        # (Batch, 4, Height / 8, Width / 8) -> (Batch, 320, Height / 8, Width / 8)
        output = self.unet(latent, context, time)
        # (Batch, 320, Height / 8, Width / 8) -> (Batch, 4, Height / 8, Width / 8)
        output = self.unet_output(output)
        return output

        
