# Script to find bland pixels for all pixels in the Machine Learning Toolkit for CRISM Analysis mineral dataset.
# Uses code and model from the Machine Learning Toolkit for CRISM Analysis: https://github.com/Banus/crism_ml/tree/master.
# Requires the mineral pixel dataset to be collated first, using the mineral_dataset_collation.py script.
# Also requires the images to be formatted in the following file structure:
# data/raw_mineral_images
# |- IMAGE_NAME (e.g. ATU00003561F)
# |  |- .img file (e.g. ATU0003561F_01_IF168L_TRR3.img)
# |  |- .lbl file (e.g. ATU0003561F_01_IF168L_TRR3.lbl)
#
# Assumes that we only need to save the 350 bands used by N2N4M and the Toolkit HBM model.
# The data is saved to the data/extracted_mineral_pixel_data dir as bland_pixels_to_match_mineral_pixels.json

import pandas as pd
import numpy as np
import os
from os.path import join, dirname

from n2n4m.io import load_image_from_shortcode
from n2n4m.postprocessing import calculate_pixel_blandness
from n2n4m.wavelengths import ALL_WAVELENGTHS, PLEBANI_WAVELENGTHS
from n2n4m.preprocessing import load_dataset


PACKAGE_DIR = dirname(os.getcwd())
DATA_DIR = join(PACKAGE_DIR, "data")  # Trunk of the data directory
IMAGE_DATA_DIR = join(DATA_DIR, "raw_mineral_images")  # Contains the raw CRISM data
COLLATED_DATA_DIR = join(
    DATA_DIR, "extracted_mineral_pixel_data"
)  # Contains the collated mineral pixel data after running the mineral_dataset_collation.py script
CRISM_ML_DIR = join(
    DATA_DIR, "CRISM_ML"
)  # Contains the datasets from the Machine Learning Toolkit for CRISM Analysis
TMP_DIR = join(
    DATA_DIR, "tmp_bland_pixel_finder_data"
)  # Temporary directory to save the bland pixel data
if not os.path.exists(TMP_DIR):
    os.mkdir(TMP_DIR)

OUTPUT_DIR = COLLATED_DATA_DIR  # Output directory for the bland pixel data

WINDOW_SIZE = 50  # The window size to use for the bland pixel calculation i.e. how many pixels to consider above/below the pixel of interest.
N_PIXELS = 3  # The number of bland pixels to find for each mineral pixel

BAND_MASK = np.isin(
    ALL_WAVELENGTHS, PLEBANI_WAVELENGTHS
)  # Mask to filter the bands to the 350 required by the Plebani model

np.seterr(all="ignore")  # Ignore floating point by zero errors

# Load the mineral pixel dataset
mineral_pixel_data = load_dataset(join(COLLATED_DATA_DIR, "mineral_pixel_dataset.json"))

# Loop through all the images in the mineral pixel dataset
for image_id in mineral_pixel_data["Image_Name"].unique():
    # Load the image
    image_id_df = pd.DataFrame({"Image_Name": [image_id]})
    image_array = load_image_from_shortcode(image_id_df, IMAGE_DATA_DIR)
    image_array = image_array[
        :, :, BAND_MASK
    ]  # Filter the bands to the 350 required by the Plebani model
    # Calculate the blandness of each pixel in the image
    pixel_blandness = calculate_pixel_blandness(
        image_array.reshape(-1, len(PLEBANI_WAVELENGTHS)),
        image_array.shape[:2],
        CRISM_ML_DIR,
    )

    bland_pixels_to_keep = []
    # Loop through all the mineral pixels in the image
    for _, mineral_pixel in mineral_pixel_data[
        mineral_pixel_data["Image_Name"] == image_id
    ].iterrows():
        # Get the coordinates
        mineral_pixel_x_coord = mineral_pixel["Coordinates"][0]
        mineral_pixel_y_coord = mineral_pixel["Coordinates"][1]

        mineral_pixel_row = mineral_pixel_y_coord - 1
        mineral_pixel_col = mineral_pixel_x_coord - 1

        # Adjust window dims to ensure we don't go out of bounds
        neg_window = min(mineral_pixel_row, WINDOW_SIZE)
        pos_window = min(image_array.shape[0] - mineral_pixel_row, WINDOW_SIZE)

        # Get the index positions of the bland pixels in the window, sorted by blandness score
        idx_best_bland_pixels = []
        sorted_idx = np.argsort(
            pixel_blandness[
                mineral_pixel_row - neg_window : mineral_pixel_row + pos_window,
                mineral_pixel_col,
            ]
        )
        sorted_idx = sorted_idx[
            ::-1
        ]  # Reverse the order to get the best bland pixels first
        for idx in sorted_idx:
            if len(idx_best_bland_pixels) >= N_PIXELS:
                break
            # Check that pixel is not the pixel of interest, and isn't a "bad" pixel (blandness score of -np.inf)
            if (
                idx - neg_window != 0
                and pixel_blandness[
                    mineral_pixel_row - neg_window + idx, mineral_pixel_col
                ]
                != -np.inf
            ):
                idx_best_bland_pixels.append(idx)

        # Get the coordinates of the best bland pixels
        # Column is just x coord, row is the index position of the pixel in the window centred around mineral_pixel_y_coord.
        # We need to subtract the neagtive window size to get the absolute row position of the window, and then add the index position of the pixel in the window.
        coords_best_bland_pixels = [
            [mineral_pixel_x_coord, mineral_pixel_y_coord - neg_window + y]
            for y in idx_best_bland_pixels
        ]

        # Add the best bland pixels to the list
        bland_pixel_spectra = []
        for coords in coords_best_bland_pixels:
            # Subtract 1 from the coords to get the 0-indexed position. Also flip the x and y coord positions as image is (rows, cols, bands).
            bland_pixel_spectra.append(image_array[coords[1] - 1, coords[0] - 1, :])

        # By averaging we decrease the noise fraction in the pixel
        average_bland_spectra = np.mean(bland_pixel_spectra, axis=0)
        average_bland_series = pd.Series(
            [
                average_bland_spectra,
                coords_best_bland_pixels,
                [mineral_pixel_x_coord, mineral_pixel_y_coord],
                image_id,
            ],
            index=[
                "Average_Spectra",
                "Bland_Pixel_Coordinates",
                "Mineral_Pixel_Coordinates",
                "Image_Name",
            ],
        )
        bland_pixels_to_keep.append(average_bland_series)

    # If no bland pixels were found, skip making a dataframe
    if len(bland_pixels_to_keep) == 0:
        continue
    bland_pixel_dataframe = pd.DataFrame(bland_pixels_to_keep)
    bland_pixel_dataframe["Image_Name"] = bland_pixel_dataframe["Image_Name"].astype(
        str
    )
    # Save each image as a separate .json in a temp dir
    bland_pixel_dataframe.to_json(join(TMP_DIR, f"{image_id}_bland_pixels.json"))

# Concatenate all the .jsons into one dataframe
bland_pixel_data = pd.concat(
    [
        pd.read_json(join(TMP_DIR, f))
        for f in os.listdir(TMP_DIR)
        if f.endswith(".json")
    ],
    ignore_index=True,
)
# Save the dataframe
bland_pixel_data.to_json(join(OUTPUT_DIR, "bland_pixels_to_match_mineral_pixels.json"))

# Clean up the temp dir
for f in os.listdir(TMP_DIR):
    os.remove(join(TMP_DIR, f))
os.rmdir(TMP_DIR)
