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
                            dtype=torch.float32,
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
        x = x.transpose(1, 2).reshape(batch_size, channels, h, w)
        return x

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, down_factor = 2):
        super(DownSample, self).__init__()

        # Convolude to reduce dimensionality
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Conv2d(in_channels, 
                out_channels // 2, 
                kernel_size=4, 
                stride = down_factor,
                padding=1
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
                in_channels, out_channels//2,
                kernel_size=4, stride=up_factor, padding=1
            ),
            nn.Conv2d(
                out_channels//2, out_channels//2 ,
                kernel_size = 1, padding = 0
            )
        )

        # Maxpool to reduce dimensionality
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels//2, kernel_size=1, padding = 0)
        )
    def forward(self, x):
        return torch.cat([self.conv(x), self.up(x)], dim=1)
    
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
        self.attn_block = nn.ModuleList(
            [SelfAttention(out_channels) for i in range(num_layers)]
        )  
        self.down_block = DownSample(out_channels, out_channels) if down_sample else nn.Identity()

        self.final_block = nn.ModuleList(
            [nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1) for i in range(num_layers)]
        )

    def forward(self, x, t_emb):
        # Forward for each layer
        for i in range(self.num_layers):
            res_input = x

            # Convoltion/Down sampling. Resnet block.
            x = self.conv1[i](x)
            x = x + self.time_embed[i](t_emb)[:, :, None, None]
            x = self.conv2[i](x)
            x = x + self.final_block[i](res_input)

            # self attention
            res_attn = self.attn_block[i](x)
            x = x + res_attn

        x = self.down_block(x)

        return x
    
class Bottleneck(nn.Module):
    def __init__(self, in_channels,  out_channels, t_emb_dim = 128, num_layers = 2):
        super(Bottleneck, self).__init__()

        self.num_layers = num_layers

        # ResNet
        self.conv1 = nn.ModuleList(
            [NormActConvNetwork(in_channels if i == 0 else out_channels, 
                                out_channels) for i in range(num_layers + 1)]
        )
        self.conv2 = nn.ModuleList(
            [NormActConvNetwork(out_channels, out_channels) for i in range(num_layers + 1)]
        )
        self.time_embed_block = nn.ModuleList(
            [TimeEmbedding(out_channels, t_emb_dim) for i in range(num_layers + 1)]
        )
        self.attn_block = nn.ModuleList(
            [SelfAttention(out_channels) for i in range(num_layers)]
        )
        self.res_block = nn.ModuleList(
            [nn.Conv2d(in_channels if i == 0 else out_channels,
                       out_channels, kernel_size=1) for i in range(num_layers + 1)]
        )
    def forward(self, x, time_embed):
        resnet_input = x

    # First-Resnet Block
        x = self.conv1[0](x)
        x = x + self.time_embed_block[0](time_embed)[:, :, None, None]
        x = self.conv2[0](x)
        x = x + self.res_block[0](resnet_input)

        # Loop of Self-Attention + Resnet Blocks
        for i in range(self.num_layers):
            # Self Attention
            res_attn = self.attn_block[i](x)
            x = x + res_attn
            
            # Resnet Block
            resnet_input = x
            x = self.conv1[i+1](x)
            x = x + self.time_embed_block[i+1](time_embed)[:, :, None, None]
            x = self.conv2[i+1](x)
            x = x + self.res_block[i+1](resnet_input)

        return x
    
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, 
                t_emb_dim = 128, num_layers = 2, up_sample= True):
        super(UpConv, self).__init__()

        self.num_layers = num_layers

        self.conv1 = nn.ModuleList(
            [NormActConvNetwork(
                in_channels if i==0 else out_channels, 
                out_channels) for i in range(num_layers)]
        )
        
        self.conv2 = nn.ModuleList(
            [NormActConvNetwork(out_channels, out_channels) for _ in range(num_layers)]
        )
        
        self.time_embed_block = nn.ModuleList(
            [TimeEmbedding(out_channels, t_emb_dim) for _ in range(num_layers)]
        )
        
        self.attn_block = nn.ModuleList(
            [SelfAttention(out_channels) for _ in range(num_layers)]
        )
        
        self.up_block = UpSample(in_channels, in_channels // 2) if up_sample else nn.Identity()
        
        self.final_block = nn.ModuleList(
            [nn.Conv2d(in_channels if i==0 else out_channels, out_channels, kernel_size=1) for i in range(num_layers)]
        )
    def forward(self, x, down_output, t_emb):
        # Down output is the skip connection between the up and down sampling portions of UNet
        
        # Upsampling
        x = self.up_block(x)
        x = torch.cat([x, down_output], dim =1)

        for i in range(self.num_layers):
            resnet_input = x

            # ResNet
            x = self.conv1[i](x)
            x = x + self.time_embed_block[i](t_emb)[:,:, None, None]
            x = self.conv2[i](x)
            x = x + self.final_block[i](resnet_input)

            # Attention
            attn = self.attn_block[i](x)
            x = x + attn

            return x

class UNet(nn.Module):
    def __init__(self, 
                 input_channels = 1, 
                 down_channels = [32, 64, 128, 256],
                 mid_channels = [256, 256, 128],
                 up_channels = [256, 128, 64, 32], 
                 down_sample = [True, True, False],
                 time_emb_dim = 128, 
                 num_downc_layers = 2, 
                 num_midc_layers = 2, 
                 num_upc_layers = 2):
        super(UNet, self).__init__()

        self.in_channels = input_channels
        self.down_ch = down_channels
        self.mid_ch = mid_channels
        self.up_ch = up_channels
        self.t_emb_dim = time_emb_dim
        self.down_sample = down_sample
        self.num_downc_layers = num_downc_layers
        self.num_midc_layers = num_midc_layers
        self.num_upc_layers = num_upc_layers

        self.up_sample = list(reversed(self.down_sample))

        # Get VAE output here

        # For now, using a simple conv because it kept breaking
        self.cv1 = nn.Conv2d(self.in_channels, self.down_ch[0], kernel_size=3, padding=1)
        
        # Time Embedding 
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim), 
            nn.SiLU(), 
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )
        
        # DownC Blocks
        self.downs = nn.ModuleList([
            DownConv(
                self.down_ch[i], 
                self.down_ch[i+1], 
                self.t_emb_dim, 
                self.num_downc_layers, 
                self.down_sample[i]
            ) for i in range(len(self.down_ch) - 1)
        ])
        
        # MidC Block
        self.mids = nn.ModuleList([
            Bottleneck(
                self.mid_ch[i], 
                self.mid_ch[i+1], 
                self.t_emb_dim, 
                self.num_midc_layers
            ) for i in range(len(self.mid_ch) - 1)
        ])
        
        # UpC Block
        self.ups = nn.ModuleList([
            UpConv(
                self.up_ch[i], 
                self.up_ch[i+1], 
                self.t_emb_dim, 
                self.num_upc_layers, 
                self.up_sample[i]
            ) for i in range(len(self.up_ch) - 1)
        ])
        
        # Final Convolution
        self.cv2 = nn.Sequential(
            nn.GroupNorm(8, self.up_ch[-1]), 
            nn.Conv2d(self.up_ch[-1], self.in_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x, t):
        
        out = self.cv1(x)
        
        # Time Projection
        t_emb = create_time_emb(t, self.t_emb_dim)
        t_emb = self.t_proj(t_emb)
        
        # DownC outputs
        down_outs = []
        
        for down in self.downs:
            down_outs.append(out)
            out = down(out, t_emb)
        
        # MidC outputs
        for mid in self.mids:
            out = mid(out, t_emb)
        
        # UpC Blocks
        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb)
            
        # Final Conv
        out = self.cv2(out)
        
        return out

if __name__ == "__main__":
    # Create a UNet model instance
    model = UNet()

    # Create a random tensor for input (Batch Size, Channels, Height, Width)
    x = torch.randn(4, 1, 32, 32)  # Batch of 4 images, 1 channel (grayscale), 32x32 resolution

    # Create a random tensor for the time steps (Batch Size,)
    t = torch.randint(0, 10, (4,))  # Random time steps for each image in the batch

    # Forward pass
    output = model(x, t)
    print(f"Output shape: {output.shape}")

    # Check output values
    print(f"Output min value: {output.min().item()}, max value: {output.max().item()}")

    # Create a dummy target and loss function for testing
    target = torch.randn_like(output)  # Random target
    loss_fn = nn.MSELoss()

    # Compute loss
    loss = loss_fn(output, target)
    print(f"Loss: {loss.item()}")

    # Perform a backward pass to ensure gradients are computed
    loss.backward()

    # Check gradients of the first layer
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Gradients for {name}: {param.grad.abs().mean().item()}")
            break  # Just checking the first parameter's gradients


