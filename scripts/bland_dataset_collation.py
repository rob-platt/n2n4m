# This is a script to collate all of the bland pixels detailed in the Machine Learning Toolkit for CRISM Analysis from the original images https://github.com/Banus/crism_ml/tree/master.
# This requires the CRISM_bland_unratioed.mat file from https://cs.iupui.edu/~mdundar/CRISM.htm. This must be in the data/CRISM_ML dir.
# This also requires the images to be formatted in the following file structure:
# data/raw_bland_images
# |- IMAGE_NAME (e.g. ATU00003561F)
# |  |- .img file (e.g. ATU0003561F_01_IF168L_TRR3.img)
# |  |- .lbl file (e.g. ATU0003561F_01_IF168L_TRR3.lbl)

# The data is saved to the data/extracted_bland_pixel_data dir as a JSON file.
# Each pixel has the following information:
# - Coordinates
# - Pixel_Class from [1]
# - Image_Name (unique 5 long hexadecimal shortcodes)
# - Spectrum (the reflectance values for each band)

# References
# 1. Plebani E, Ehlmann BL, Leask EK, Fox VK, Dundar MM.
# A machine learning toolkit for CRISM image analysis. Icarus. 2022 Apr;376:114849.


import pandas as pd
import numpy as np
import os
from os.path import join, dirname
import scipy.io
from time import time

from n2n4m.io import load_image

PARENT_DIR = dirname(os.getcwd())
DATA_DIR = join(PARENT_DIR, "data")
IMAGE_DATA_DIR = join(DATA_DIR, "raw_bland_images")
CRISM_ML_DIR = join(DATA_DIR, "CRISM_ML")
TMP_DIR = join(DATA_DIR, "tmp_bland_pixel_data")
OUTPUT_DIR = join(DATA_DIR, "extracted_bland_pixel_data")

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

if not os.path.exists(TMP_DIR):
    os.mkdir(TMP_DIR)

# Load the Bland_Pixel dataset from [1].
try:
    bland_pixel_mat = scipy.io.loadmat(join(CRISM_ML_DIR, "CRISM_bland_unratioed.mat"))
except:
    raise IOError(f"Could not load the CRISM_bland_unratioed.mat from {CRISM_ML_DIR}")
# Get the image acquisition names from the dict.
mat_image_names = [image[0][0].strip(".mat") for image in bland_pixel_mat["im_names"]]


# Get the image name for each pixel in the .mat dataset. Initially are numerically encoded, so also decode. Image names given as the unique 5 long hexadecimal shortcodes (e.g. 3561F)
image_name_for_pixel_list = bland_pixel_mat["pixims"]
image_name_for_pixel_list = np.array(
    [mat_image_names[pix_im[0] - 1] for pix_im in image_name_for_pixel_list]
)  # Original file is matlab, which is 1-indexed, python is 0-indexed so -1 to index position to account for that

# List the names of all the images in the directory
downloaded_images_list = os.listdir(IMAGE_DATA_DIR)

if len(downloaded_images_list) == 0:
    raise IOError(f"No images found in directory: {IMAGE_DATA_DIR}")


total_time_taken = 0
# Loop through all of the available images
for idx, image_name in enumerate(downloaded_images_list):
    spectra_list = []
    coords_list = []
    image_name_list = []
    pixel_class_list = []
    start_time = time()
    # Check if the current image (slicing for the shortcode) contains bland pixels in the .mat file
    if image_name[-5:] not in mat_image_names:
        print(f"Skipping {image_name} as not in .mat file")
        continue

    image_dir = join(IMAGE_DATA_DIR, image_name)
    image_files_list = os.listdir(image_dir)
    # Identify the actual .img file
    image_filename = [
        filename
        for filename in image_files_list
        if "trr3.img" in filename[-8:] or "TRR3.img" in filename[-8:]
    ]
    if len(image_filename) == 1:
        image_filename = image_filename[0]
    else:
        print(f"Skipping {image_name} as no/multiple image file(s) found")
        continue

    image_path = join(image_dir, image_filename)
    image_array, hdr = load_image(image_path)
    image_shape = image_array.shape[:2]  # Spatial dims only

    # Mask of the bland pixels in this image, from the list of all the bland pixels across all the images
    relevant_pixel_mask = np.isin(image_name_for_pixel_list, image_name[-5:])
    # Get the pixel coordinates of all the bland pixels in this image
    relevant_pixel_coords = bland_pixel_mat["pixcrds"][relevant_pixel_mask]

    # Loop through coords of bland pixels in that image, slice the image array to get the spectra for that pixel, and add all the information to the lists.
    for coords in relevant_pixel_coords:
        x, y = coords
        # Some of the pixel coordinates are [0,0] which is invalid - a) Matlab is 1 indexed, b) often there are multiple pixels attributed to the same [0,0] which isn't possible.
        if x == 0 or y == 0 or x > image_shape[1] or y > image_shape[0]:
            continue
        spectra_list.append(
            image_array[y - 1, x - 1]
        )  # image_array is (n_rows, n_cols, n_bands) so flipped x and y. -1 offset to account for python indexing starting at 0 but coordinates starting at (1, 1).
        coords_list.append(coords)
        image_name_list.append(image_name[-5:])  # Hexadecimal shortcode only
        pixel_class_list.append([39])  # Bland pixels are class 39

    # Create a dataframe of that images pixels and save it to a JSON file in a temporary dir
    image_dataframe = pd.DataFrame(
        {
            "Coordinates": coords_list,
            "Pixel_Class": pixel_class_list,
            "Image_Name": image_name_list,
            "Spectrum": spectra_list,
        }
    )
    image_dataframe.to_json(f"{join(TMP_DIR, image_name)}.json")

    finish_time = time()
    total_time_taken += finish_time - start_time
    print(
        f"Finished {image_name} in {finish_time-start_time} seconds, projected completion time: {(((total_time_taken/(idx+1))*(len(downloaded_images_list)-(1+idx)))/60) + 5} minutes",
        flush=True,
    )

# Now collate all the individual jsons into a single dataset.
image_json_path_list = os.listdir(TMP_DIR)
image_dataframe_list = []
for filepath in image_json_path_list:
    image_dataframe_list.append(
        pd.read_json(join(TMP_DIR, filepath), dtype={"Image_Name": "string"})
    )

full_dataset = pd.concat(image_dataframe_list, ignore_index=True)
full_dataset.to_json(join(OUTPUT_DIR, "bland_pixel_dataset.json"))
print(
    f"Finished collating all the images into a single dataset, saved to {join(OUTPUT_DIR, 'bland_pixel_dataset.json')}",
    flush=True,
)

# Clean up the temporary directory
for filepath in image_json_path_list:
    os.remove(join(TMP_DIR, filepath))
os.rmdir(TMP_DIR)
