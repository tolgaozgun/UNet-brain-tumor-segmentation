# Import image from PIL
from PIL import Image
import numpy as np


def rescale_image(image):
    # Find the minimum and maximum values in the image

    min_val = np.min(image)
    max_val = np.max(image)

    # Scale the image to the range of 0 to 255
    scaled_image = (image - min_val) * (255.0 / (max_val - min_val))

    scaled_image = scaled_image.astype(np.uint8)
    
    return scaled_image

# Read out.jpg
img = Image.open("out.jpg")

img_array = np.asarray(img)

print(img_array.shape)

# Print min max and unique values of img_array
print("Min max")
print(img_array.min(), img_array.max())
print("Unique values")
print(np.unique(img_array))

# Load image from /Users/tolgaozgun/UNet-brain-tumor-segmentation/data/masks/sub-01_slice14_mask.png
img2 = Image.open("/Users/tolgaozgun/UNet-brain-tumor-segmentation/data/masks/sub-01_slice14_mask.png")


# Print min max and unique values of this img2
print("Min max2")
print(np.asarray(img2).min(), np.asarray(img2).max())
print("Unique values2")
print(np.unique(np.asarray(img2)))




# Rescale min max values to 0-255
img = rescale_image(img)

# Save numpy array as image
img = Image.fromarray(img)



# Save image as out.png
img.save("out1.png")

