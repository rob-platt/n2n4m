import pytest
import pandas as pd
import numpy as np

import n2n4m.utils as utils


def test_label_list_to_string():
    test_df = pd.DataFrame(
        {"Pixel_Class": [[1], [2], [3]], "Image_Name": ["test1", "test2", "test3"]}
    )
    test_df_converted = utils.label_list_to_string(test_df)

    assert type(test_df_converted) == pd.DataFrame
    assert isinstance(test_df_converted.loc[0, "Pixel_Class"], np.integer)
    assert test_df.shape == test_df_converted.shape


def test_label_string_to_list():
    test_df = pd.DataFrame(
        {"Pixel_Class": [1, 2, 3], "Image_Name": ["test1", "test2", "test3"]}
    )
    test_df_converted = utils.label_string_to_list(test_df)

    assert type(test_df_converted) == pd.DataFrame
    assert type(test_df_converted.loc[0, "Pixel_Class"]) == list
    assert test_df.shape == test_df_converted.shape


def test_convert_coordinates_to_xy():
    test_df = pd.DataFrame(
        {
            "Coordinates": [[1, 2], [3, 4], [5, 6]],
            "Image_Name": ["test1", "test2", "test3"],
        }
    )
    test_df_converted = utils.convert_coordinates_to_xy(test_df)

    assert type(test_df_converted) == pd.DataFrame
    assert test_df_converted.shape == (3, 3)  # additional column as x, y
    assert "Coordinates" not in test_df_converted.columns
    assert "x" in test_df_converted.columns
    assert "y" in test_df_converted.columns
    assert test_df_converted.loc[0, "x"] == 1
    assert test_df_converted.loc[0, "y"] == 2


def test_convert_xy_to_coordinates():
    test_df = pd.DataFrame(
        {"x": [1, 3, 5], "y": [2, 4, 6], "Image_Name": ["test1", "test2", "test3"]}
    )
    test_df_converted = utils.convert_xy_to_coordinates(test_df)

    assert type(test_df_converted) == pd.DataFrame
    # removed one column as x, y now coordinates
    assert test_df_converted.shape == (3,2)  
    assert "x" not in test_df_converted.columns
    assert "y" not in test_df_converted.columns
    assert "Coordinates" in test_df_converted.columns
    assert test_df_converted.loc[0, "Coordinates"] == [1, 2]
