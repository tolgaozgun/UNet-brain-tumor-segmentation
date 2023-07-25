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

VALIDATION_PERCENTAGE = 0.2

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

def resize_image(image: np.array):
    return image.resize((256, 256))

    

def parse_images(flair_imgs, t1w_imgs, t2w_imgs, sub_no):
    assert(t1w_imgs.shape[2] == t2w_imgs.shape[2] == flair_imgs.shape[2])
    no_of_slices = t1w_imgs.shape[2]

    for i in range(0, no_of_slices):
        t1w_img = t1w_imgs[..., i]
        t2w_img = t2w_imgs[..., i]
        flair_img = flair_imgs[..., i]

        # Resize images to 256x256
        t1w_img = resize_image(t1w_img)
        t2w_img = resize_image(t2w_img)
        flair_img = resize_image(flair_img)


        file_name = f"{sub_no}_seq{i}_fake_B.png"
        t1c_path = os.path.join(os.getcwd(), "t1c", file_name)

        if os.path.exists(t1c_path):
            print(f"T1c image does not exist: {t1c_path}")
            continue

        # Load image from t1c_path as grayscale
        t1c_img = Image.open(t1c_path).convert('L')

        t1c_img = np.asarray(t1c_img)

        t1c_img = rescale_image(t1c_img)

        assert(t1w_img.shape == t2w_img.shape == flair_img.shape)



        concatenated_img = np.concatenate([t1w_img[..., np.newaxis], t2w_img[..., np.newaxis], flair_img[..., np.newaxis], t1c_img], axis=-1)

        concatenated_img = rescale_image(concatenated_img)

        img_A_data = Image.fromarray(concatenated_img)
            # CycleGAN validation dataset

        img_A_data.save(os.path.join(IMAGE_OUTPUT, f"{sub_no}_slice{i}.png"))


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

def main():
    image_folder_path = os.path.join(DATASET_FOLDER, "sourcedata")

    data_folders = get_valid_image_folders(image_folder_path)

    for index, data_folder in enumerate(data_folders):
        flair_imgs, t1w_imgs, t2w_imgs = load_image_from_folder(data_folder)
        base_name = os.path.basename(data_folder)
        parse_images(flair_imgs, t1w_imgs, t2w_imgs, base_name)
        print(f"Image for {base_name} completed. Total count: {index}")


if __name__ == "__main__":
    main()

