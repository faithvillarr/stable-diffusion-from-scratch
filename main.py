import torch
from models.text_encoder import TextEncoder
from models.unet import UNet
from models.vae import VAE
from diffusion.forward_diffusion import forward_diffusion
from diffusion.backward_diffusion import reverse_diffusion
from diffusion.scheduler import NoiseScheduler

def run_stable_diffusion(text_prompt, image_size=(256, 256)):
    # Initialize models
    text_encoder = TextEncoder()
    vae = VAE()
    unet = UNet()
    scheduler = NoiseScheduler()

    # Encode text prompt
    text_embeddings = text_encoder.encode(text_prompt)

    # Sample random latent space (image size)
    latent = torch.randn(1, 4, image_size[0] // 8, image_size[1] // 8)

    # Apply forward diffusion (add noise)
    noisy_latent = forward_diffusion(latent, scheduler.timesteps)

    # Reverse diffusion (denoise the latent space)
    denoised_latent = reverse_diffusion(noisy_latent, unet, scheduler.timesteps)

    # Decode latent space back to image
    generated_image = vae.decode(denoised_latent)

    return generated_image

if __name__ == "__main__":
    prompt = "A black cat."
    run_stable_diffusion(prompt)