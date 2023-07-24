import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from nilearn import plotting

# Load image from data/imgs/sub-01_slice14.png
img = Image.open("data/imgs/sub-07_slice9.png")

# Load mask from x.png
mask = Image.open("9.png")

# mask = np.asarray(mask)

# Print min max and unique values of mask
# print("Min max")
# print(mask.min(), mask.max())

mask = np.asarray(mask)
# Print unique values of mask
print("Unique values")
unique, count = np.unique(mask, return_counts=True)
print(f"Unique values: {unique}")
print(f"Counts: {count}")

# Print frequency of values in mask


plt.figure()
plt.subplot(1,2,1)
plt.imshow(mask, cmap='hot', alpha=1.0)
plt.subplot(1,2,2)
plt.imshow(img, cmap='gray')
# plt.imshow(mask, cmap='hot', alpha=0.5)
# plt.subplot(1,1,1)
# plt.imshow(mask, cmap='hot', alpha=0.5, interpolation='none')
plt.show()

# Normalize the mask to 0-255
# mask = np.asarray(mask)

# mask = (mask - mask.min()) * (255.0 / (mask.max() - mask.min()))

# mask = mask.astype(np.uint8)


# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(mask, cmap='hot', alpha=0.5)
# plt.subplot(1,2,2)
# plt.imshow(img, cmap='gray')
# plt.imshow(mask, cmap='hot', alpha=0.5)
# # plt.subplot(1,1,1)
# # plt.imshow(mask, cmap='hot', alpha=0.5, interpolation='none')
# plt.show()

# # Print min max and unique values of mask
# print("Min max")
# print(mask.min(), mask.max())


