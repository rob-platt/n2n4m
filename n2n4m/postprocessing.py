import pandas as pd
import numpy as np
import os
import n2n4m.io
from crism_ml.io import image_shape
from crism_ml.preprocessing import filter_bad_pixels, remove_spikes_column, replace
from crism_ml.train import train_model_bland, feat_masks, compute_bland_scores
from n2n4m.wavelengths import ALL_WAVELENGTHS, PLEBANI_WAVELENGTHS
from n2n4m.utils import convert_coordinates_to_xy, convert_xy_to_coordinates


def load_image_from_shortcode(
    mineral_sample: pd.DataFrame,
    data_dir: str = "../data/raw_images",
) -> dict:
    """
    Given a mineral sample from a dataset, load the image.

    Parameters
    ----------
    mineral_sample : pd.DataFrame
        Data sample, must contain "Image_Name" column.
    data_dir : str, optional
        Directory where the raw images are stored. Assumes each image is in a separate folder.

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
    image = n2n4m.io.load_image(image_filename)
    return image


def calculate_pixel_blandness(
    spectra: np.ndarray,
    im_shape: tuple,
    train_set_dir: str = "data",
) -> np.ndarray:
    """
    Calculates the blandness of each pixel in an image using the GMM model from Plebani et al. 2022
    https://github.com/Banus/crism_ml/tree/master

    Parameters
    ----------
    image : ndarray
        Contains the spectral data.
        If ndarray, must be the spectral data only, in shape (n_spectra, n_bands).
        Should contain only the 350 channels required by the Plebani model.
    im_shape : tuple
        The shape of the image.
    train_set_dir : str
        Path to the directory containing the bland pixel training set.
        Default is "data".

    Returns
    -------
    blandness : np.ndarray
        Array of blandness values for each pixel in the image in shape (n_rows, n_cols).
    """
    # Need these checks as Plebani functions will silent fail otherwise.
    if type(spectra) != np.ndarray:
        raise TypeError(f"spectra must be a dictionary or numpy array, not {type(spectra)}")
    if spectra.shape[1] != 350: 
        raise ValueError(
            f"spectra must have 350 bands, not {spectra.shape[1]}. Use the PLEBANI_WAVELENGTHS constant to filter the bands."
        )
    fin0, fin = feat_masks()  # fin0 for bland pixels, fin for non-bland pixels
    bland_model = (  # Train bland model using the unratioed bland pixel dataset
        train_model_bland(train_set_dir, fin0)
    )
    spectra, bad_pixel_mask = filter_bad_pixels(spectra)

    despiked_spectra = remove_spikes_column(  # Remove spikes using a median filter with window size 3, removing spikes larger than 5 std dev. Calculated per column
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
