import pytest
import pandas as pd
import numpy as np

import n2n4m.postprocessing as postprocessing
from n2n4m.io import load_image
from crism_ml.io import image_shape
from n2n4m.preprocessing import ALL_WAVELENGTHS, PLEBANI_WAVELENGTHS

BAND_MASK = [True if band in PLEBANI_WAVELENGTHS else False for band in ALL_WAVELENGTHS]


@pytest.fixture
def crism_image():
    return load_image("tests/test_postprocessing/3561F/ATU0003561F_01_IF168L_TRR3.img")


def test_calculate_pixel_blandness(crism_image):
    image_array, hdr = crism_image # Unpack image and header
    im_shape = image_array.shape[:-1] # Get spatial dimensions of image
    spectra = image_array.reshape(-1, image_array.shape[-1])[:, BAND_MASK] # Flatten image and mask bands not used in Plebani model

    blandness_scores = postprocessing.calculate_pixel_blandness(
        spectra, im_shape=im_shape
    )
    assert type(blandness_scores) == np.ndarray
    assert blandness_scores.shape == im_shape
    assert (
        blandness_scores[0, 0] == -np.inf
    )  # Corner pixels are always bad pixels, so should be -np.inf
    pytest.raises(
        ValueError, postprocessing.calculate_pixel_blandness, spectra[:100], im_shape
    )
    pytest.raises(
        TypeError, postprocessing.calculate_pixel_blandness, im_shape, im_shape
    )
