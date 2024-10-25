# UNet 
# Take a noisy image at timestep t and return the predicted noise. 

'''
Things to Implement:
[ ] Down-Conv Blocks
[ ] Up- Conv Blocks
[ ] Bottleneck Blocks
[ ] Attention
[ ] Skip Connections between Up and Down Sampling Blocks
[ ] Time-Step embedding for input

'''

# Import relevant libraries
print("Importing pandas...")
import pandas as pd
print("Importing numpy...")
import numpy
print("Importing torch libraries...")
import torch 
import torch.nn as nn
import torchvision

# Time Step Embedding
def create_time_emb(time_step, emb_dim):
    # Create a tensor of size emb_dim that use sinusoidal positional
    #   encoding to capture a given time_step.
    if emb_dim % 2 != 0:
        return ValueError("Embedding dimension not divisible by 2.")

    # Create factor
    factor = 2 * torch.arange(start = 0, 
                            end = emb_dim // 2, 
                            dtype=torch.float32
                            ) / (emb_dim)

    # Exponentiate the 10000
    factor = 10000 ** factor

    time_emb = time_step[:,None] / factor # (B) > (B, t_emb_dim // 2)

    # Concat sin and cos of the factor 
    time_emb = torch.cat([torch.sin(time_emb), torch.cos(time_emb)], dim=1) # (B , t_emb_dim)

    return time_emb

class NormActConvNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups = 8, kernel_size = 3):
        super(NormActConvNetwork, self).__init__()

        # Group Normalization on batch
        self.g_norm = nn.GroupNorm(num_groups, in_channels)

        # Sigmoid Linear Unit Activation
        self.silu = nn.SiLU()

        # Convolution
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding = (kernel_size - 1) // 2)

    def forward(self, x):
        x = self.g_norm(x)
        x = self.silu(x)
        x = self.conv(x)
        return x

class TimeEmbedding(nn.Module):
    def __init__(self, out_dim, t_emb_dim):
        super(TimeEmbedding, self).__init__()

        # Time Embedding Block
        self.time_embed = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(t_emb_dim, out_dim)
        )

    def forward(self, x):
        return self.time_embed(x)
    
class SelfAttention(nn.Module):
    def __init__(self, num_channels, num_groups = 8, num_heads = 4):
        super(SelfAttention, self).__init__()

        # Group Normalizations
        self.g_norm = nn.GroupNorm(num_groups, num_channels)

        # Self-Attetion
        self.attn = nn.MultiheadAttention(num_channels, num_heads, batch_first=True)

    def forward(self, x):
        batch_size, channels, h, w = x.shape
        x = x.reshape(batch_size, channels, h*w)

        x = self.g_norm(x)
        x = x.transpose(1, 2)
        x, _ = self.attn(x, x, x) # query, key and value are all the same

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, down_factor = 2):
        super(DownSample, self).__init__()

        # Convolude to reduce dimensionality
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Conv2d(in_channels, 
                out_channels // 2, 
                kernel_size=4, 
                stride = down_factor
            )
        )

        # Maxpool to reduce dimensionality
        self.mpool = nn.Sequential(
            nn.MaxPool2d(down_factor, down_factor),
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0)
        )
    def forward(self, x):
        return torch.cat([self.conv(x), self.mpool(x)], dim=1)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, up_factor = 2):
        super(UpSample, self).__init__()

        # Convolude to reduce dimensionality
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4, stride=up_factor, padding=1
            ),
            nn.Conv2d(
                out_channels//2, out_channels//2 ,
                kernel_size = 1
            )
        )

        # Maxpool to reduce dimensionality
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels//2, kernel_size=1)
        )
    def forward(self, x):
        return torch.cat([self.conv(x), self.mpool(x)], dim=1)
    
class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, 
                t_emb_dim = 128, num_layers = 2, 
                down_sample = True):
        super(DownConv, self).__init__()

        self.num_layers = num_layers

        self.conv1 = nn.ModuleList(
            [NormActConvNetwork(
                in_channels if i == 0 else out_channels, 
                out_channels) 
                for i in range(num_layers)]
        )
        self.conv2 = nn.ModuleList([
            NormActConvNetwork(out_channels, out_channels) for i in range(num_layers)
        ])

        self.time_embed = nn.ModuleList([
            TimeEmbedding(out_channels, t_emb_dim) for i in range(num_layers)
        ])
