import torch
from PIL import Image
from models.text_encoder import TextEncoder
from models.unet import UNet
from models.vae import VAE
from diffusion.forward_diffusion import forward_diffusion
from diffusion.backward_diffusion import backward_diffusion
from diffusion.scheduler import NoiseScheduler
import numpy as np

def run_stable_diffusion(text_prompt, image_size=(256, 256)):
    # Initialize models
    print("Instantiating models...")
    text_encoder = TextEncoder()
    vae = VAE()
    unet = UNet()
    scheduler = NoiseScheduler()

    # Encode text prompt
    print("Creating text embeddings...")
    text_embeddings = text_encoder.encode(text_prompt)

    # Load the image using PIL
    print("Grabbing image...")

    # Load the image and convert to RGB if not already
    og_image = Image.open("training_examples/sonny-angel.png").convert("RGB")
    # Resize or crop if necessary to match the model’s input size
    og_image = og_image.resize((256, 256))

    # Convert the image to a NumPy array, then to a tensor, and add the batch dimension
    og_image = torch.tensor(np.array(og_image)).float().permute(2, 0, 1).unsqueeze(0) / 255.0  # Normalize to [0, 1]

    # Sample random latent space (image size)
    print("Running VAE...")
    latent = vae.encode(og_image)
    unlatent = vae.decode(latent)

    display_images(og_image, unlatent)

def display_images(original_image, processed_image):
    # Convert processed image tensor back to PIL Image for display
    processed_image = processed_image.squeeze(0).permute(1, 2, 0).numpy()  # Change shape to (H, W, C)
    processed_image = (processed_image * 255).astype(np.uint8)  # Convert to uint8 for PIL

    # Create a figure to display images
    plt.figure(figsize=(10, 5))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")

    # Processed Image
    plt.subplot(1, 2, 2)
    plt.imshow(processed_image)
    plt.title("Processed Image")
    plt.axis("off")

    plt.show()
    

    # # Apply forward diffusion (add noise)
    # noisy_latent = forward_diffusion(latent, scheduler.timesteps)

    # # Reverse diffusion (denoise the latent space)
    # denoised_latent = backward_diffusion(noisy_latent, unet, scheduler.timesteps)

    # # Decode latent space back to image
    # generated_image = vae.decode(denoised_latent)

    # return generated_image

if __name__ == "__main__":
    prompt = "A black cat."
    run_stable_diffusion(prompt)
    print("I ran!")