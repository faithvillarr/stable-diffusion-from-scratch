# Stable Diffusion from Scratch

**Goal:** Create and train a stable diffusion model using just PyTorch and OpenAI's CLIP model for text embedding. 

## Core Components
### The Variational Autoencoder
[**General Idea**]The Variational Autoencoder or VAE is a model which maps to a latent space which is fed further into the diffusion process. The encoder takes a tensor of a 256px x 256px image and turns it into a latent vector. This vector, after being processed by the rest of the model, is fed into the decoder to retrieve the final outputted image.  

![image](https://github.com/user-attachments/assets/91bef06f-9a32-4d7a-ad57-058a3351568a)

The fun part of a VAE the '**variational**' aspect. When returning a letent space from an inputted image, the model actually returns two vectors that are used as a mean and variance in a normal distribution. By selecting from this distribution, randomness and variety is added to the output of the model, allowing for the model to prepare to process a variety of inputs. 

#### Encoder and Decoder Notation

The VAE model has two parts:

1. **The Encoder** $$(q_\phi(z|x))$$: Encodes an image $$x$$ into a latent representation $$z$$. The encoder is parameterized by $$\phi$$, which are its weights. 


$$q_\phi(z|x) = \mathcal{N}(z| \mu_\phi(x), \sigma_\phi(x)^2)$$


   where $$\mu_\phi(x)$$ and $$\sigma_\phi(x)$$ are the mean and variance respectively predicted by the encoder network.

2. **The Decoder** $$(p_\theta(x|z))$$: Reconstructs the image $$x$$ from the latent representation $$z$$. The decoder is parameterized by $$\theta$$, which are its weights.

$$ p_\theta(x|z) = \mathcal{N}(x | \mu_\theta(z), \sigma_\theta(z)^2)$$

#### The Reparameterization Trick

The random nature of selecting an output from a distribution presents an issue when faced with the idea of back-propogating through the network. 

Keeping in mind the shape of our distribution definition:

$$\mathcal{N}(z| \mu_\phi(x), \sigma_\phi(x)^2)$$

The solution that has been to use our variable $$z$$ as a deterministic function of $$x$$ and assign the randomness to a random noise variable $$\epsilon$$ such that we represent $$z$$ as:

$$z=\mu_\phi\(x\)+ \sigma _\phi \(x\)\odot \epsilon \text{ where } \epsilon \sim \mathcal{N}(0, 1)$$

This result is now a differentiable function of $$\phi\(x\)$$ and $$\sigma _\phi(x)$$. This means we can use backpropogation again in our network! Problem solved! :)


#### The Loss Function

The ideal likelihood function would approximate the likelihood of getting our original data $$x$$ given our produced data $$z$$ and model parameters $$\phi$$. The function would look as follows:

$$p(x|\theta)=\int p(x|z)p(z)dz$$

where $$p(z)$$ is the prior distribution of the latent variable $$z$$ given the disstribution is typically $$\mathcal{N}(0, 1)$$.

Since we cannot integrate this equation, we instead **minimize the  negative log-likelihood** rather than **maximize the likelihood**.

### The U-Net: A Noise Predicting Model

In the Stable Diffusion pipeline, the **U-Net** is used to predict the noise present in an image during the reverse diffusion process. As we iteratively remove noise, the U-Net learns to adjust its predictions in line with the characteristics of the image being generated.

The core idea of the U-Net architecture is a **downsampling and upsampling pathway** that allows for both **local** and **global context** extraction, making it particularly effective for generating high-quality images with fine-grained details. 

#### Downsampling Path

This part of the U-Net reduces the spatial resolution, capturing the image's global structure. It consists of several convolutional layers with pooling operations that reduce the image’s dimensions. The output of each stage is known as a “feature map,” which captures patterns in the image at various resolutions.

#### Upsampling Path

After reaching the lowest resolution in the downsampling path, the network performs a **series of upsampling operations**. This pathway reconstructs the spatial dimensions of the image, adding back the details. Importantly, the U-Net **concatenates the feature maps** from the downsampling pathway to each corresponding layer in the upsampling path. This skip connection mechanism gives the model access to high-resolution details from the input image.

The final layer in the U-Net outputs a **noise prediction** for each pixel in the latent representation. This noise prediction is used to iteratively denoise the image as it progresses through the diffusion process. 

#### U-Net Notation and Diffusion Process

Let:
- $$x_t$$ represent the noisy image at time step $$t$$.
- $$\epsilon_\theta(x_t, t) $$ be the U-Net’s predicted noise at time step $$t$$.
  
The model denoises by iteratively subtracting the predicted noise from each image at step $$t$$, following:

$$x_{t-1} = x_t - \epsilon_\theta(x_t, t)$$

This iterative process, combined with the VAE’s latent space, allows for controlled, gradual refinement of the image, enhancing coherence and detail with each step. 

#### Diffusion Loss

To guide the U-Net in predicting the correct noise at each time step, the model minimizes a **mean squared error (MSE) loss** between the true noise $$\epsilon$$ and the predicted noise $$\epsilon_\theta(x_t, t)$$:

$$L_{\text{diffusion}} = \mathbb{E}_{x, \epsilon, t}\[ \Vert \epsilon - \epsilon _{\theta} (x_t, t) \Vert ^2 \]$$

This loss function trains the U-Net to accurately predict the noise, making it effective at denoising the image across the diffusion process.

### The Full Diffusion Process

The Stable Diffusion model combines the **VAE** and **U-Net** to create a generative process that can start from a pure noise vector and iteratively denoise it into a coherent image. The **encoder** maps the image into a latent space, where noise is injected, and the **decoder** reconstructs the image from the latent vector produced by the U-Net. 

1. **Encode**: An input image $$x$$ is encoded into a latent vector $$z$$ by the VAE encoder.
2. **Diffuse**: Noise is iteratively added to $$z$$, reaching a point where the image is unrecognizable.
3. **Reverse Diffusion**: The U-Net takes over, predicting and removing noise at each step until a clean latent vector is recovered.
4. **Decode**: The VAE decoder then reconstructs the final image from the denoised latent vector, generating the model’s output.


