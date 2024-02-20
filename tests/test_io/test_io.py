import pytest
import numpy as np
import pandas as pd

import n2n4m.io

@pytest.fixture
def crism_image():
    return n2n4m.io.load_image("tests/test_postprocessing/3561F/ATU0003561F_01_IF168L_TRR3.img")


def test_load_images_from_shortcode():
    good_mineral_sample = pd.DataFrame({"Image_Name": "3561F", "Pixel_Class": [10]})
    bad_mineral_sample = pd.DataFrame({"Image_Name": "AAAAA", "Pixel_Class": [10]})
    good_data_dir = "tests/test_postprocessing"
    bad_data_dir = "tests/"
    image_array = n2n4m.io.load_image_from_shortcode(good_mineral_sample, good_data_dir)
    assert type(image_array) == np.ndarray
    pytest.raises(
        ValueError,
        n2n4m.io.load_image_from_shortcode,
        good_mineral_sample,
        bad_data_dir,
    )  # Check Value Error raised if directory for image doesn't exist
    pytest.raises(
        ValueError,
        n2n4m.io.load_image_from_shortcode,
        bad_mineral_sample,
        good_data_dir,
    )  # Check Value Error raised if image file doesn't exist