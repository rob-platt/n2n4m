import pandas as pd
import numpy as np
import os
from crism_ml.io import load_image, image_shape
from crism_ml.preprocessing import filter_bad_pixels, remove_spikes_column, replace
from crism_ml.train import train_model_bland, feat_masks, compute_bland_scores
from n2n4m.preprocessing import ALL_WAVELENGTHS, PLEBANI_WAVELENGTHS


def convert_coordinates_to_xy(dataframe):
    if type(dataframe) == pd.Series:
        dataframe["x"] = dataframe["Coordinates"][0]
        dataframe["y"] = dataframe["Coordinates"][1]
        dataframe = dataframe.drop("Coordinates")
    else:
        dataframe["x"] = dataframe["Coordinates"].apply(lambda x: x[0])
        dataframe["y"] = dataframe["Coordinates"].apply(lambda x: x[1])
        dataframe = dataframe.drop(columns=["Coordinates"])
    return dataframe


def convert_xy_to_coordinates(dataframe):
    if type(dataframe) == pd.Series:
        dataframe["Coordinates"] = [dataframe["x"], dataframe["y"]]
        dataframe = dataframe.drop(index=["x", "y"])
    else:
        dataframe["Coordinates"] = dataframe.apply(lambda x: [x["x"], x["y"]], axis=1)
        dataframe = dataframe.drop(columns=["x", "y"])
    return dataframe


def load_image_from_shortcode(
    mineral_sample, data_dir="../data/raw_images", load_all_bands=False
):
    """
    Given a mineral sample from the Plebani dataset, load the image.
    Uses modified load_image() function from Plebani et al. 2022.
    https://github.com/Banus/crism_ml/tree/master


    Parameters
    ----------
    mineral_sample : pd.DataFrame
        Sample from the Plebani datset, must contain "Image_Name" column.
    data_dir : str, optional
        Directory where the raw images are stored. Assumes each image is in a separate folder.
    load_all_bands : bool, optional
        Whether to return all bands or only the bands in the Plebani dataset.
        The default is False.

    Returns
    -------
    image : dict
        Dictionary of image data.
    """
    image_shortcode = mineral_sample["Image_Name"].values[0]
    image_folder_list = os.listdir(data_dir)
    image_folder = [folder for folder in image_folder_list if image_shortcode in folder]
    if len(image_folder) == 0 or len(image_folder) > 1:
        raise ValueError(
            f"Image folder not found or multiple folders found for shortcode {image_shortcode}."
        )
    image_folder = os.path.join(data_dir, image_folder[0])
    image_filename = [
        filename for filename in os.listdir(image_folder) if "3.img" == filename[-5:]
    ]  # Requires TRR3 processing
    if len(image_filename) == 0 or len(image_filename) > 1:
        raise ValueError(
            f"Image file not found or multiple files found for shortcode {image_shortcode}."
        )
    image_filename = os.path.join(image_folder, image_filename[0])
    image = load_image(image_filename)

    if load_all_bands:
        return image

    band_mask = [
        True if band in PLEBANI_WAVELENGTHS else False for band in ALL_WAVELENGTHS
    ]  # Boolean mask of whether each band in the L sensor is included in the Plebani dataset
    image["IF"] = image["IF"][:, band_mask]
    return image


def calculate_pixel_blandness(
    image,
    train_set_dir="/home/rp1818/CRISM_image_de-aging/CRISM/Plebani_Dataset",
    im_shape=None,
):
    """
    Calculates the blandness of each pixel in an image using the GMM and code from Plebani et al. 2022
    https://github.com/Banus/crism_ml/tree/master

    Parameters
    ----------
    image : dict, ndarray
        Contains the spectral data. If dict, must be in format given by crism_ml.io.load_image().
        If ndarray, must be the spectral data only, in shape (n_spectra, n_bands).S
        Should contain only the 350 channels required by the Plebani model.
    train_set_dir : str
        Path to the directory containing the bland pixel training set.
    im_shape : tuple, optional
        The shape of the image. Only required if image is a numpy array.

    Returns
    -------
    blandness : np.ndarray
        Array of blandness values for each pixel in the image in shape (n_rows, n_cols).
    """
    if type(image) == dict:
        spectra = image["IF"]
        im_shape = image_shape(image)
    elif type(image) == np.ndarray:
        if im_shape is None:
            raise ValueError("If image is a numpy array, im_shape must be provided.")
        spectra = image
    else:
        raise TypeError(f"Image must be a dictionary or numpy array, not {type(image)}")

    fin0, fin = feat_masks()  # fin0 for bland pixels, fin for non-bland pixels
    bland_model = (  # Train bland model using the unratioed bland pixel dataset
        train_model_bland(train_set_dir, fin0)
    )
    spectra, bad_pixel_mask = filter_bad_pixels(spectra)
    print(
        f"There are {bad_pixel_mask.sum()} bad pixels in the image of {spectra.shape[0]} pixels"
    )
    despiked_spectra = remove_spikes_column(  # Remove spikes using a median filter with window size 3, removing spikes larger than 5 std dev. calculated per column
        spectra.reshape(*im_shape, -1), size=3, sigma=5
    ).reshape(
        spectra.shape
    )
    bland_scores = compute_bland_scores(  # Compute the blandness score for each pixel
        despiked_spectra, (bland_model, fin0)
    )
    screened_bland_scores = replace(  # replace the blandness score of any bad pixel by -inf, so that when it comes to bland pixel selection, it will not be selected.
        bland_scores, bad_pixel_mask, -np.inf
    ).reshape(
        im_shape
    )
    return screened_bland_scores
