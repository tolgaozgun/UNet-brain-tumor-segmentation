import os
import shutil
import nibabel as nib
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# MASKS_DIR = "/Users/tolgaozgun/Downloads/segmentation/"
MASKS_DIR = "workspace/shared-datas/TurkBeyinProjesi/GaziBrains_BIDS/GAZI_BRAINS_2020/derivatives/segmentation"

MASK_OUTPUT = "../data/masks_new"

limit_classes = True

use_four_sequences = False

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
        # print(mask_img.min(), mask_img.max())

        # mask_img = rescale_image(mask_img)
        mask_img = mask_img.astype(np.uint8)

        if limit_classes:

            if use_four_sequences:
                # 0,2,3 -> 0 background
                # 6 -> 5 iskemik
                # 7 -> 2 peritumor
                # 8 -> 3 contrast enhanced
                # 9,21 -> 1 necrosis
                # 16 -> 6 kavernom
                # 11,13,14 -> 7 hemorrage
                # 1,4,5,10,12,15,17,18,19,20,22,23,24 -> 4 roi
                pass
            
            else:
                # 0,2,3 -> 0 background
                # 1,10,12,15,17,18,19,20,22,23,24 -> 1 roi
                # 4 -> 2 lateral
                # 5 -> 3 third
                # 6 -> 4 iskemik
                # 7,8,9,11,13,14,16,21 -> 5 tumor
                # Use the mapping above to map the number values of mask_img
                mask_img = np.where(mask_img == 0, 0, mask_img)
                mask_img = np.where(mask_img == 2, 0, mask_img)
                mask_img = np.where(mask_img == 3, 0, mask_img)
                mask_img = np.where(mask_img == 1, 1, mask_img)
                mask_img = np.where(mask_img == 10, 1, mask_img)
                mask_img = np.where(mask_img == 12, 1, mask_img)
                mask_img = np.where(mask_img == 15, 1, mask_img)
                mask_img = np.where(mask_img == 17, 1, mask_img)
                mask_img = np.where(mask_img == 18, 1, mask_img)
                mask_img = np.where(mask_img == 19, 1, mask_img)
                mask_img = np.where(mask_img == 20, 1, mask_img)
                mask_img = np.where(mask_img == 22, 1, mask_img)
                mask_img = np.where(mask_img == 23, 1, mask_img)
                mask_img = np.where(mask_img == 24, 1, mask_img)
                mask_img = np.where(mask_img == 4, 2, mask_img)
                mask_img = np.where(mask_img == 5, 3, mask_img)
                mask_img = np.where(mask_img == 6, 4, mask_img)
                mask_img = np.where(mask_img == 7, 5, mask_img)
                mask_img = np.where(mask_img == 8, 5, mask_img)
                mask_img = np.where(mask_img == 9, 5, mask_img)
                mask_img = np.where(mask_img == 11, 5, mask_img)
                mask_img = np.where(mask_img == 13, 5, mask_img)
                mask_img = np.where(mask_img == 14, 5, mask_img)
                mask_img = np.where(mask_img == 16, 5, mask_img)
                mask_img = np.where(mask_img == 21, 5, mask_img)

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

