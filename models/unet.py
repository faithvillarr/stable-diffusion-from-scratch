import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.conv = UNetBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x_down = self.pool(x)
        return x_down, x

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = UNetBlock(in_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        x = torch.cat((x, skip_connection), dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super(UNet, self).__init__()

        # Contracting path
        self.down1 = DownSample(in_channels, base_channels)
        self.down2 = DownSample(base_channels, base_channels * 2)
        self.down3 = DownSample(base_channels * 2, base_channels * 4)
        self.down4 = DownSample(base_channels * 4, base_channels * 8)

        # Bottleneck
        self.bottleneck = UNetBlock(base_channels * 8, base_channels * 16)

        # Expansive path
        self.up1 = UpSample(base_channels * 16, base_channels * 8)
        self.up2 = UpSample(base_channels * 8, base_channels * 4)
        self.up3 = UpSample(base_channels * 4, base_channels * 2)
        self.up4 = UpSample(base_channels * 2, base_channels)

        # Output layer
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Contracting path
        x1_down, x1 = self.down1(x)
        x2_down, x2 = self.down2(x1_down)
        x3_down, x3 = self.down3(x2_down)
        x4_down, x4 = self.down4(x3_down)

        # Bottleneck
        x_bottleneck = self.bottleneck(x4_down)

        # Expansive path
        x_up1 = self.up1(x_bottleneck, x4)
        x_up2 = self.up2(x_up1, x3)
        x_up3 = self.up3(x_up2, x2)
        x_up4 = self.up4(x_up3, x1)

        # Final output
        output = self.final_conv(x_up4)
        return output

# Example usage:
if __name__ == "__main__":
    model = UNet()
    input_image = torch.randn(1, 3, 256, 256) 
    output_image = model(input_image)
    print(output_image.shape)  
