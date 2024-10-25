'''
This is the first file I wrote so there are considerable notes on 
the function of the network and just how to use PyTorch.
'''

import torch
import torch.nn as nn
from torchvision import transforms

class VAE(nn.Module):
    # Here we define the layers in our network.
    def __init__(self): 
        super(VAE, self).__init__()
        
        #  Encoder Network Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        
        self.flatten = nn.Flatten() # for imputting to FC layers

        # mean and var networks. Fully connected.
        self.fc_mu = nn.Linear(32 * 32 * 64, 128)
        self.fc_var = nn.Linear(32 * 32 * 64, 128)

        # Decoder Network Layers
        self.fc_decode = nn.Linear(128, 64 * 32 * 32) 
        self.unconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.unconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.unconv3 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Activation FUncitons
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, input_tensor): # A 'forward' pass. Where we apply the init layers. 
        # Take an input tensor x and compress the features into 
        # a smaller latent representation. 
        # Return the mean and variance for selecting from distribution

        # Apply our conv2d
        latent_representation = self.relu(self.conv1(input_tensor))
        latent_representation = self.relu(self.conv2(latent_representation))
        latent_representation = self.relu(self.conv3(latent_representation))

        # Flatten for FC layers
        latent_representation = self.flatten(latent_representation)

        # Get mu and var for dsitribution trick
        mean = self.fc_mu(latent_representation)
        var = self.fc_var(latent_representation)

        return mean, var

    def reparameterize(self, mu, var):
        # Use mean and var to select from distribution
        std_dev = torch.exp(0.5 * var)  
        epsilon = torch.randn_like(std_dev)  

        # Return random sample from distribution. Yay diversity!   
        return mu + epsilon * std_dev          

    def decode(self, encoded_tensor):
        # Forward pass through the decoder
        res_tensor = self.relu(self.fc_decode(encoded_tensor))

        # Realign with output shape
        res_tensor = res_tensor.view(-1, 64, 32, 32)

        # Reverse earlier filters
        res_tensor = self.relu(self.unconv1(res_tensor))
        res_tensor = self.relu(self.unconv2(res_tensor))
        res_tensor = self.sigmoid(self.unconv3(res_tensor))

        return res_tensor # the reconstructed image

    def forward(self, input_tensor):
        mean, var = self.encode(input_tensor)
        sample_dist = self.reparameterize(mean, var)
        return self.decode(sample_dist)
    
if __name__ == "__main__":
    vae = VAE()
    to_pil = transforms.ToPILImage()

    # Generate a random image
    random_image = torch.randn(1, 3, 256, 256)
    print(f"Original Image Shape: {random_image.shape}")
    to_pil(random_image.squeeze(0)).show()

    reconstructed_image = vae(random_image)
    print(f"Reconstructed Image Shape: {reconstructed_image.shape}")

    reconstructed_image = reconstructed_image.squeeze(0)
    pil_image = to_pil(reconstructed_image)
    pil_image.show()
