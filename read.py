import nibabel as nib
import numpy as np
from PIL import Image

# Read a nii.gz file from /Users/tolgaozgun/Downloads/segmentation/sub-01/anat/sub-01_dseg.nii.gz
file = nib.load("/Users/tolgaozgun/Downloads/segmentation/sub-01/anat/sub-01_dseg.nii.gz").get_fdata()

# Print minimum and maximum values
print(file.min(), file.max())

# Print shape
print(file.shape)

one_file = file[..., 13]

# Save array of columns between 256-270 into array.txt
np.savetxt("array.txt", one_file[:, 256:270])

one_file = one_file.astype(np.uint8)

# Save array of columns between 256-270 into array.txt
np.savetxt("array1.txt", one_file[:, 256:270])

# Save the file as a png
img = Image.fromarray(one_file)

# Print min and max values from img
print(img.getextrema())

# Save img as img.png
img.save("img.png")


