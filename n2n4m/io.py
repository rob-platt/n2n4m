import numpy as np
import os
from spectral.io import envi, spyfile

import crism_ml.io


def crism_to_mat(fname: str, flatten: bool = False) -> dict:
    """Convert a CRISM ENVI image to a Matlab-like dictionary.
    Modified from https://github.com/Banus/crism_ml/blob/master/crism_ml/io.py
    Modification loads all CRISM L bands (0-438) instead of the original reduced set.

    Loads an ENVI image as a Matlab-like dictionary with spectra (IF) and pixel
    coordinates (x, y). If the header (.hdr) is not found, it is automatically
    generated from a .lbl file (using the approach in the
    `CRISM spectral calculator`_); if neither is available, an error is raised.

    .. _CRISM spectral calculator: https://github.com/jlaura/crism/blob/\
        master/csas.py

    Parameters
    ----------
    fname: str
        ENVI file to open (.hdr or .img)
    flatten: bool
        flatten an image array to (npix, nchan) and saves the coordinates to
        the x,y fields (default: False)

    Returns
    -------
    mat: dict
        a dictionary storing the spectra and the pixels coordinates (if flatten
        is True)
    """
    # pylint: disable=import-outside-toplevel

    band_select = np.arange(0, 438, 1)
    fbase, _ = os.path.splitext(fname)
    try:
        img = envi.open(f"{fbase}.hdr")
    except spyfile.FileNotFoundError:
        crism_ml.io._generate_envi_header(f"{fbase}.lbl")
        img = envi.open(f"{fbase}.hdr")

    arr = img.load()

    mdict = {"IF": arr[:, :, band_select]}
    if flatten:  # use coordinate arrays for indexing
        xx_, yy_ = np.meshgrid(np.arange(arr.shape[1]), np.arange(arr.shape[0]))
        mdict.update({"x": xx_.ravel() + 1, "y": yy_.ravel() + 1})
        mdict["IF"] = mdict["IF"].reshape((-1, len(band_select)))

    return mdict


def load_image(fname: str) -> dict:
    """Try to load a .mat file and fall back to ENVI if not found.
    
    Parameters
    ----------
    fname : str
        Path to file to load.

    Returns
    -------
    image : dict
        Dictionary of image data.
    """
    try:
        return crism_ml.io.loadmat(fname)
    except (FileNotFoundError, NotImplementedError, ValueError):
        return crism_to_mat(fname, flatten=True)
