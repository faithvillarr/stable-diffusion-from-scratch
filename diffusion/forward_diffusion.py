import torch

def forward_diffusion(latent, timesteps):
    # Apply forward noise to the latent representation over 'timesteps'
    noisy_latent = latent + noise
    return noisy_latent
