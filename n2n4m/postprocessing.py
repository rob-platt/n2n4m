import numpy as np
import os

from crism_ml.preprocessing import filter_bad_pixels, remove_spikes_column, replace
from crism_ml.train import train_model_bland, feat_masks, compute_bland_scores


def check_data_exists(filepath: str) -> bool:
    """
    Check if Plebani bland pixel model training data exists.

    Parameters
    ----------
    filepath : str
        Path to the data directory.

    Returns
    -------
    bool
        True if the file exists, False otherwise.
    """
    return os.path.exists(os.path.join(filepath, "CRISM_bland_unratioed.mat"))


def calculate_pixel_blandness(
    spectra: np.ndarray,
    im_shape: tuple[int, int],
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
    im_shape : tuple[int, int]
        The spatial shape of the image.
    train_set_dir : str
        Path to the directory containing the bland pixel training set.
        Default is "data".

    Returns
    -------
    blandness : np.ndarray
        Array of blandness values for each pixel in the image in shape (n_rows, n_cols).
    """
    if check_data_exists(train_set_dir) == False:
        raise FileNotFoundError(
            f"Training data not found in {train_set_dir}. Please download the training data from"
            f"https://zenodo.org/records/13338091/files/CRISM_bland_unratioed.mat and place in the data directory."
        )
    # Need these checks as Plebani functions will silent fail otherwise.
    if type(spectra) != np.ndarray:
        raise TypeError(
            f"spectra must be a dictionary or numpy array, not {type(spectra)}"
        )
    if spectra.shape[1] != 350:
        raise ValueError(
            f"spectra must have 350 bands, not {spectra.shape[1]}. Use the PLEBANI_WAVELENGTHS constant to filter the bands."
        )
    fin0, fin = feat_masks()  # fin0 for bland pixels, fin for non-bland pixels
    bland_model = (  # Train bland model using the unratioed bland pixel dataset
        train_model_bland(train_set_dir, fin0)
    )
    filtered_spectra, bad_pixel_mask = filter_bad_pixels(spectra, copy=True)

    despiked_spectra = remove_spikes_column(  # Remove spikes using a median filter with window size 3, removing spikes larger than 5 std dev. Calculated per column
        filtered_spectra.reshape(*im_shape, -1), size=3, sigma=5
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
