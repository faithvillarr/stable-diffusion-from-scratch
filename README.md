# Stable Diffusion from Scratch

**Goal:** Create and train a stable diffusion model using just PyTorch and OpenAI's CLIP model for text embedding. 

## Core Components
#### The Variational Autoencoder
The Variational Autoencoder or VAE is a model which maps to a latent space which is fed further into the diffusion process. The encoder takes a tensor of a 256px x 256px image and turns it into a latent vector. This vector, after being processed by the rest of the model, is fed into the decoder to retrieve the final outputted image. 

The *fun* part of a VAE the 'variational' aspect. When returning a letent space from an inputted image, the model actually returns two vectors that are used as a mean and variance in a normal distribution. By selecting from this distribution, variety is added to the output of the model, allowing for the model to prepare to process a variety of inputs.  

![image](https://github.com/user-attachments/assets/91bef06f-9a32-4d7a-ad57-058a3351568a)

#### The U-Net: A Noise Predicting Model
The 

