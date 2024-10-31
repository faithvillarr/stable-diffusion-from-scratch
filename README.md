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


#### The U-Net: A Noise Predicting Model
The attempt

