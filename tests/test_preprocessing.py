import pytest
import pandas as pd

from n2n4m.wavelengths import ALL_WAVELENGTHS, PLEBANI_WAVELENGTHS
import n2n4m.preprocessing as preprocessing


@pytest.fixture
def raw_data():
    # Small sample dataset with columns [Coordinates(list), Image_Name(str), Pixel_Class, Spectrum(list)].
    data = pd.read_json("tests/sample_data.json", dtype={"Image_Name": "string"})
    return data

def test_load_dataset(raw_data):
    sample_data_load = preprocessing.load_dataset("tests/sample_data.json")
    assert sample_data_load.equals(raw_data)
    pytest.raises(FileNotFoundError, preprocessing.load_dataset, "tests/not_a_file.json")
    pytest.raises(ValueError, preprocessing.load_dataset, "tests/test_preprocessing.py")

@pytest.fixture
def expanded_data():
    # Same sample as raw_data, with spectrum column expanded to 1 column per band.
    data = pd.read_json("tests/sample_data_expanded.json", dtype={"Image_Name": "string"})
    return data

def test_expand_spectrum(raw_data, expanded_data):
    sample_data_expanded = preprocessing.expand_dataset(raw_data)
    assert len(sample_data_expanded.columns) == len(expanded_data.columns)
    assert len(sample_data_expanded) == len(expanded_data)

def test_drop_bad_bands(expanded_data):
    cleaned_data = preprocessing.drop_bad_bands(expanded_data, PLEBANI_WAVELENGTHS)
    assert len(cleaned_data.columns) == len(PLEBANI_WAVELENGTHS) + 3 # 3 extra columns for coordinates, image name, and pixel class
    assert len(cleaned_data) == len(expanded_data) 
    assert all([str(band) in cleaned_data.columns for band in PLEBANI_WAVELENGTHS])
    assert all([str(band) not in cleaned_data.columns for band in ALL_WAVELENGTHS if band not in PLEBANI_WAVELENGTHS])


