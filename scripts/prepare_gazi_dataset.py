import os
import shutil
import nibabel as nib
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# DATASET_FOLDER = "/Users/tolgaozgun/gazi_brains_2020/data/GAZI_BRAINS_2020/sourcedata"
DATASET_FOLDER = "/workspace/shared-datas/TurkBeyinProjesi/GaziBrains_BIDS/GAZI_BRAINS_2020/"


# derivatives/segmentation/sub-xx/anat/sub-xx_dseg.nii.gz
IMAGE_OUTPUT = "../data/imgs/"
MASK_OUTPUT = "../data/masks/"

VALIDATION_PERCENTAGE = 0.2

# cyclegan_trainA_folder = os.path.join(GAZI_CYCLEGAN_OUTPUT, "trainA")
# cyclegan_trainB_folder = os.path.join(GAZI_CYCLEGAN_OUTPUT, "trainB")
# cyclegan_valA_folder = os.path.join(GAZI_CYCLEGAN_OUTPUT, "valA")
# cyclegan_valB_folder = os.path.join(GAZI_CYCLEGAN_OUTPUT, "valB")
# pix2pix_trainA_folder = os.path.join(GAZI_PIX2PIX_OUTPUT, "A", "train")
# pix2pix_trainB_folder = os.path.join(GAZI_PIX2PIX_OUTPUT, "B", "train")
# pix2pix_valA_folder = os.path.join(GAZI_PIX2PIX_OUTPUT, "A", "val")
# pix2pix_valB_folder = os.path.join(GAZI_PIX2PIX_OUTPUT, "B", "val")

# # Create trainA and trainB directories if they don't exist
# os.makedirs(cyclegan_trainA_folder, exist_ok=True)
# os.makedirs(cyclegan_trainB_folder, exist_ok=True)
# os.makedirs(cyclegan_valA_folder, exist_ok=True)
# os.makedirs(cyclegan_valB_folder, exist_ok=True)
# os.makedirs(pix2pix_trainA_folder, exist_ok=True)
# os.makedirs(pix2pix_trainB_folder, exist_ok=True)
# os.makedirs(pix2pix_valA_folder, exist_ok=True)
# os.makedirs(pix2pix_valB_folder, exist_ok=True)

def load_image_from_folder(folder_path):

    base_name = os.path.basename(folder_path)
    folder_path = os.path.join(folder_path, "anat")
    
    # Load mandatory files
    t1w_path = os.path.join(folder_path, f"{base_name}_T1w.nii.gz")
    flair_path = os.path.join(folder_path, f"{base_name}_FLAIR.nii.gz")
    t2w_path = os.path.join(folder_path, f"{base_name}_T2w.nii.gz")
    t1w_img = nib.load(t1w_path).get_fdata()
    flair_img = nib.load(flair_path).get_fdata()
    t2w_img = nib.load(t2w_path).get_fdata()

    # Load optional files if available
    # gadolinium_t1w_path = os.path.join(folder_path, f"{base_name}_ce-GADOLINIUM_T1w.nii.gz")
    # gadolinium_t1w_img = nib.load(gadolinium_t1w_path).get_fdata()


    return flair_img, t1w_img, t2w_img



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
    

def parse_images(flair_imgs, t1w_imgs, t2w_imgs, sub_no):
    parse_counter = 0

    assert(t1w_imgs.shape[2] == t2w_imgs.shape[2] == flair_imgs.shape[2])
    no_of_slices = t1w_imgs.shape[2]

    for i in range(0, no_of_slices):
        t1w_img = t1w_imgs[..., i]
        t2w_img = t2w_imgs[..., i]
        flair_img = flair_imgs[..., i]

        assert(t1w_img.shape == t2w_img.shape == flair_img.shape)

        concatenated_img = np.concatenate([t1w_img[..., np.newaxis], t2w_img[..., np.newaxis], flair_img[..., np.newaxis]], axis=-1)

        concatenated_img = rescale_image(concatenated_img)

        img_A_data = Image.fromarray(concatenated_img)
            # CycleGAN validation dataset

        img_A_data.save(os.path.join(IMAGE_OUTPUT, f"{sub_no}_slice{i}.png"))

    parse_counter += 1
    print(f"Image for {sub_no} completed. Total count: {parse_counter}")

def parse_masks(mask_imgs, sub_no):
    parse_counter = 0
    no_of_slices = mask_imgs.shape[2]

    for i in range(0, no_of_slices):
        mask_img = mask_imgs[..., i]

        img_B_data = Image.fromarray(mask_img)
            # CycleGAN validation dataset

        img_B_data.save(os.path.join(MASK_OUTPUT, f"{sub_no}_slice{i}_mask.png"))

    global parse_counter
    parse_counter += 1
    print(f"Mask for {sub_no} completed. Total count: {parse_counter}")


def rescale_image(image):
    # Find the minimum and maximum values in the image

    min_val = np.min(image)
    max_val = np.max(image)

    # Scale the image to the range of 0 to 255
    scaled_image = (image - min_val) * (255.0 / (max_val - min_val))

    scaled_image = scaled_image.astype(np.uint8)
    
    return scaled_image

def get_valid_image_folders(root_dir):
    data_folders = []
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        anat_path = os.path.join(folder_path, "anat")
        if os.path.isdir(folder_path) and os.path.isdir(anat_path):
            data_folders.append(folder_path)

    return data_folders

def get_valid_mask_folders(root_dir):
    data_folders = []
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        anat_path = os.path.join(folder_path, "anat")
        if os.path.isdir(folder_path) and os.path.isdir(anat_path):
            data_folders.append(folder_path)

    return data_folders

def main():
    image_folder_path = os.path.join(DATASET_FOLDER, "sourcedata")
    mask_folder_path = os.path.join(DATASET_FOLDER, "derivatives", "segmentation")

    data_folders = get_valid_image_folders(image_folder_path)
    mask_folders = get_valid_mask_folders(mask_folder_path)

    for data_folder in data_folders:
        flair_imgs, t1w_imgs, t2w_imgs = load_image_from_folder(data_folder)
        base_name = os.path.basename(data_folder)
        parse_images(flair_imgs, t1w_imgs, t2w_imgs, base_name)
    
    for mask_folder in mask_folders: 
        mask_imgs = load_mask_from_folder(mask_folder)
        base_name = os.path.basename(mask_folder)
        parse_masks(mask_imgs, base_name)


if __name__ == "__main__":
    main()

