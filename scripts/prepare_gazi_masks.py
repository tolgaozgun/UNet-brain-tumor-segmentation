import os
import shutil
import nibabel as nib
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

MASKS_DIR = "/Users/tolgaozgun/Downloads/segmentation"

MASK_OUTPUT = "../data/masks"

if not os.path.exists(MASK_OUTPUT):
    os.makedirs(MASK_OUTPUT)


def load_mask_from_folder(folder_path):

    base_name = os.path.basename(folder_path)
    folder_path = os.path.join(folder_path, "anat")
    
    # Load mandatory files
    mask_path = os.path.join(folder_path, f"{base_name}_dseg.nii.gz")
    mask_img = nib.load(mask_path).get_fdata()

    # Load optional files if available
    # gadolinium_t1w_path = os.path.join(folder_path, f"{base_name}_ce-GADOLINIUM_T1w.nii.gz")
    # gadolinium_t1w_img = nib.load(gadolinium_t1w_path).get_fdata()


    return mask_img
    
def parse_masks(mask_imgs, sub_no):
    no_of_slices = mask_imgs.shape[2]

    for i in range(0, no_of_slices):
        mask_img = mask_imgs[..., i]

        # Print min max of mask_img
        print(mask_img.min(), mask_img.max())

        # mask_img = rescale_image(mask_img)
        mask_img = mask_img.astype(np.uint8)

        # Print min max of mask_img
        print(mask_img.min(), mask_img.max())

        img_B_data = Image.fromarray(mask_img)
            # CycleGAN validation dataset

        img_B_data.save(os.path.join(MASK_OUTPUT, f"{sub_no}_slice{i}_mask.png"))

def get_valid_mask_folders(root_dir):
    data_folders = []
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        anat_path = os.path.join(folder_path, "anat")
        if os.path.isdir(folder_path) and os.path.isdir(anat_path):
            data_folders.append(folder_path)

    return data_folders

def main():
    mask_folders = get_valid_mask_folders(MASKS_DIR)

    for index, mask_folder in enumerate(mask_folders): 
        mask_imgs = load_mask_from_folder(mask_folder)
        base_name = os.path.basename(mask_folder)
        parse_masks(mask_imgs, base_name)
        print(f"Mask for {base_name} completed. Total count: {index}")


if __name__ == "__main__":
    main()

