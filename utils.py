from PIL import Image
import numpy as np

def save_image(tensor, file_name='generated_image.png'):
    image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    pil_image = Image.fromarray(image)
    pil_image.save(file_name)
