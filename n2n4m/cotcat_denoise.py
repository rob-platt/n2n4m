import numpy as np
import scipy.ndimage
import sys

import n2n4m.utils as utils


SLIDING_WINDOW_SIZE = 11
FILTER_SIZE = 5


def sharpening_median_filter(spectra: np.ndarray) -> np.ndarray:
    """
    Apply a sharpening median filter to spectra.
    As detailed in [1].

    Parameters
    ----------
    spectra : np.ndarray
        The spectra to be filtered.
        Shape: (n_rows, n_cols, n_bands)

    Returns
    -------
    filtered_spectra : np.ndarray
        The filtered spectra.
        Shape: (n_rows, n_cols, n_bands)

    References
    ----------
    Bultel B, Quantin C, Lozac'h L. Description of CoTCAT (Complement to CRISM Analysis Toolkit). 
    IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing. 
    2015 Jun;8(6):3039-49. 
    """
    mid_point_idx = SLIDING_WINDOW_SIZE//2 # The index of the middle element of the sliding window
    median_lower_bound_idx = mid_point_idx - (FILTER_SIZE//2) # The lower bound index of the median filter
    median_upper_bound_idx = mid_point_idx + (FILTER_SIZE//2) + 1 # The upper bound index of the median filter
    
    if spectra.ndim == 1:
        spectra = spectra[np.newaxis, np.newaxis, :]
    sliding_view = np.lib.stride_tricks.sliding_window_view(spectra, (1, 1, 11))
    sliding_view = sliding_view.squeeze()
    if sliding_view.ndim == 2: # If only a single spectra is passed, add spatial axes.
        sliding_view = sliding_view[np.newaxis,  np.newaxis, :, :]
    
    lower_quartiles = np.quantile(sliding_view, 0.25, axis=3)
    upper_quartiles = np.quantile(sliding_view, 0.75, axis=3)
    iqr = upper_quartiles - lower_quartiles
    lower_fence = lower_quartiles - 1.5 * iqr
    upper_fence = upper_quartiles + 1.5 * iqr

    # Where the middle (position) value of the sliding view is less than the lower fence or greater than the upper fence
    # associated with that sliding view, replace it with the median of the surrounding 4 values.
    filtered_values = np.where((sliding_view[:,:,:,mid_point_idx] < lower_fence) | (sliding_view[:,:,:,mid_point_idx] > upper_fence),
                            np.median(sliding_view[:,:,:,median_lower_bound_idx:median_upper_bound_idx],
                                        axis=3), 
                            sliding_view[:,:,:,mid_point_idx])
    # Replace the middle (position) value of the sliding view with the filtered value. Leave the edge spectrels as is.
    filtered_spectra = spectra.copy()
    filtered_spectra[:,:,mid_point_idx:-mid_point_idx] = filtered_values

    return filtered_spectra

def moving_median_filter(spectra: np.ndarray) -> np.ndarray:
    """
    Apply a moving median filter across an image.

    Parameters
    ----------
    spectra : np.ndarray
        The spectra to be filtered.
        Shape: (n_rows, n_cols, n_bands)

    Returns
    -------
    filtered_spectra : np.ndarray
        The filtered spectra.
        Shape: (n_rows, n_cols, n_bands)
    """
    lower_bound_idx = FILTER_SIZE//2
    upper_bound_idx = FILTER_SIZE//2

    if spectra.ndim == 1: # If only a single spectra is passed, add spatial axes
        spectra = spectra[np.newaxis, np.newaxis, :]

    spectral_median = scipy.ndimage.median_filter(spectra, size=(1, 1, FILTER_SIZE))
    
    filtered_spectra = spectra.copy()
    filtered_spectra[:,:,lower_bound_idx:-upper_bound_idx] = spectral_median[:,:,lower_bound_idx:-upper_bound_idx]
    return filtered_spectra

def moving_mean_filter(spectra: np.ndarray) -> np.ndarray:
    """
    Apply a moving mean filter across an image.

    Parameters
    ----------
    spectra : np.ndarray
        The spectra to be filtered.
        Shape: (n_rows, n_cols, n_bands)

    Returns
    -------
    filtered_spectra : np.ndarray
        The filtered spectra.
        Shape: (n_rows, n_cols, n_bands)
    """
    lower_bound_idx = FILTER_SIZE//2
    upper_bound_idx = FILTER_SIZE//2

    if spectra.ndim == 1: # If only a single spectra is passed, add spatial axes
        spectra = spectra[np.newaxis, np.newaxis, :]

    spectral_mean = scipy.ndimage.convolve1d(spectra, weights=np.ones(FILTER_SIZE)/FILTER_SIZE, axis=-1)
    
    filtered_spectra = spectra.copy()
    filtered_spectra[:,:,lower_bound_idx:-upper_bound_idx] = spectral_mean[:,:,lower_bound_idx:-upper_bound_idx]
    return filtered_spectra

def cotcat_denoise(spectra: np.ndarray, wavelengths: tuple[float, ...]) -> np.ndarray:
    """
    Apply CoTCAT denoising to spectra.
    Method as detailed in Bultel et al. 2015

    Parameters
    ----------
    spectra : np.ndarray
        The spectra to be filtered.
        Shape: (n_rows, n_cols, n_bands)
    wavelengths : np.ndarray
        The wavelengths of the spectra. To identify any breakpoints (gaps) in continuous spectrum.
        Shape: (n_bands,)

    Returns
    -------
    filtered_spectra : np.ndarray
        The filtered spectra.
        Shape: (n_rows, n_cols, n_bands)
    """
    breakpoint_idx = utils.find_breakpoint(wavelengths=wavelengths, threshold=0.01)
    if breakpoint_idx == -0:
        filtered_spectra = moving_mean_filter(moving_median_filter(sharpening_median_filter(spectra)))
    elif len(wavelengths) - breakpoint_idx < 11 or breakpoint_idx < 11:
        raise ValueError("The breakpoint is too close to the end of the spectra to apply CoTCAT denoising.")
    else:
        filtered_spectra = np.append(moving_mean_filter(moving_median_filter(sharpening_median_filter(spectra[:, :, :breakpoint_idx]))), 
                                    (moving_mean_filter(moving_median_filter(sharpening_median_filter(spectra[:, :, breakpoint_idx:])))), axis=2)
    return filtered_spectra