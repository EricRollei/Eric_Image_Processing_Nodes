import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def add_film_grain(image, grain_strength=0.05):
    """Add realistic film grain to image"""
    noise = np.random.normal(0, grain_strength, image.shape)
    
    # Add spatial correlation
    kernel = cv2.getGaussianKernel(3, 0.5)
    kernel = kernel @ kernel.T
    
    for i in range(image.shape[2]):
        noise[:, :, i] = cv2.filter2D(noise[:, :, i], -1, kernel)
    
    # Intensity-dependent noise
    intensity_factor = 1.0 + 0.5 * (image / 255.0)
    noise = noise * intensity_factor
    
    noisy = image + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    return noisy

# Process all images
clean_dir = Path("my_training_data/clean")
noisy_dir = Path("my_training_data/noisy")
noisy_dir.mkdir(exist_ok=True)

for img_path in tqdm(list(clean_dir.glob("*.png"))):
    img = cv2.imread(str(img_path))
    noisy = add_film_grain(img, grain_strength=15.0)  # Adjust strength as needed
    cv2.imwrite(str(noisy_dir / img_path.name), noisy)

print("✓ Synthetic grain added to all images!")