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


def test_load_images_from_shortcode():
    good_mineral_sample = pd.DataFrame({"Image_Name": "3561F", "Pixel_Class": [10]})
    bad_mineral_sample = pd.DataFrame({"Image_Name": "AAAAA", "Pixel_Class": [10]})
    good_data_dir = "tests/test_postprocessing"
    bad_data_dir = "tests/"
    image = postprocessing.load_image_from_shortcode(good_mineral_sample, good_data_dir)
    assert type(image) == dict
    pytest.raises(
        ValueError,
        postprocessing.load_image_from_shortcode,
        good_mineral_sample,
        bad_data_dir,
    )  # Check Value Error raised if directory for image doesn't exist
    pytest.raises(
        ValueError,
        postprocessing.load_image_from_shortcode,
        bad_mineral_sample,
        good_data_dir,
    )  # Check Value Error raised if image file doesn't exist


def test_calculate_pixel_blandness(crism_image):
    im_shape = image_shape(crism_image)
    spectra = crism_image["IF"][:, BAND_MASK]
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
