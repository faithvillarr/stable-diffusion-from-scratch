def backward_diffusion(noisy_latent, unet, timesteps):
    # Denoise the latent space representation step by step using the UNet
    for t in reversed(range(timesteps)):
        noisy_latent = unet(noisy_latent, t)
    return noisy_latent
