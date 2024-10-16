import numpy as np
from PIL import Image

def load_image_in_array(filepath):
    image = Image.open(filepath)
    image_array = np.asarray(image)
    return image_array

# Takes image as ndarray
def add_gausian_noise(image_array):
    row, col, ch = image_array.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noised_image = image_array + gauss
    # Keep values with [1, 255] for rgb
    noised_image = np.clip(noised_image, 0, 255).astype(np.uint8)    
    return noised_image

if __name__ == "__main__":
    img_array = load_image_in_array('sonny-angel.png')
    
    image = Image.fromarray(img_array)
    image.show()
    print('Adding noise ...')
    for i in range(0, 100):
        img_array = add_gausian_noise(img_array)

    image = Image.fromarray(img_array)
    image.show()
    print('Noise added!')
