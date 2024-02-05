# Module to run N2N denoising across an entire image. Should allow for CLI usage. Input trained model as instance of Noise2Noise1D class, with weights loaded, image, and output numpy array of denoised image, with extra bands inserted back from original image.
import numpy as np


from n2n4m.model_functions import predict
from n2n4m.model import Noise2Noise1D
from n2n4m.io import load_image
from n2n4m.wavelengths import ALL_WAVELENGTHS, PLEBANI_WAVELENGTHS


def band_index_mask(bands_to_keep: tuple = PLEBANI_WAVELENGTHS) -> tuple[np.ndarray, np.ndarray]:
    """Create a mask for the indices of bands to keep out of all CRISM L sensor bands.

    Parameters
    ----------
    bands_to_keep : tuple, optional
        The wavelengths of bands to keep.
        Default PLEBANI_WAVELENGTHS

    Returns
    -------
    include_bands_indices : np.ndarray
        The indices of the bands to keep.
    exclude_bands_indices : np.ndarray
        The indices of the bands to exclude.
    """
    band_mask = np.isin(ALL_WAVELENGTHS, bands_to_keep)
    include_bands_indices = np.where(band_mask)[0]
    exclude_bands_indices = np.where(~band_mask)[0]
    return include_bands_indices, exclude_bands_indices


def clip_bands(spectra: np.ndarray, bands_to_keep: tuple = PLEBANI_WAVELENGTHS) -> tuple[np.ndarray, np.ndarray]:
    """Clip the bands of the spectra to the bands specified in bands_to_keep. Returns the clipped spectra and the outside bands.

    Parameters
    ----------
    spectra : np.ndarray
        The spectra to clip.
    bands_to_keep : tuple, optional
        The bands to keep.
        Default PLEBANI_WAVELENGTHS

    Returns
    -------

    """

    return np.zeros((1, 1)), np.zeros((1, 1))  # Placeholder


def denoise_image(model: Noise2Noise1D, image_filepath: str) -> np.ndarray:
    image = load_image(image_filepath)
    
    return np.zeros((1, 1))  # Placeholder