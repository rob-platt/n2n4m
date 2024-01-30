import pytest
import pandas as pd

from n2n4m.wavelengths import ALL_WAVELENGTHS, PLEBANI_WAVELENGTHS
import n2n4m.preprocessing as preprocessing


@pytest.fixture
def raw_data():
    # Small sample dataset with columns [Coordinates(list), Image_Name(str), Pixel_Class, Spectrum(list)].
    data = pd.read_json("tests/sample_data_v2.json", dtype={"Image_Name": "string"})
    return data

def test_load_dataset(raw_data):
    sample_data_load = preprocessing.load_dataset("tests/sample_data_v2.json")
    assert sample_data_load.equals(raw_data)
    pytest.raises(FileNotFoundError, preprocessing.load_dataset, "tests/not_a_file.json")
    pytest.raises(ValueError, preprocessing.load_dataset, "tests/test_preprocessing.py")

@pytest.fixture
def expanded_data():
    # Same sample as raw_data, with spectrum column expanded to 1 column per band.
    data = pd.read_json("tests/sample_data_v2_expanded.json", dtype={"Image_Name": "string"})
    return data

def test_expand_spectrum(raw_data, expanded_data):
    bands = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    sample_data_expanded = preprocessing.expand_dataset(raw_data, bands=bands)
    assert len(sample_data_expanded.columns) == len(expanded_data.columns)
    assert len(sample_data_expanded) == len(expanded_data)

def test_drop_bad_bands(expanded_data):
    dummy_good_bands = [2, 3, 4, 5, 6, 7, 8, 9] # bands 0 and 1 are bad in dummy example
    cleaned_data = preprocessing.drop_bad_bands(expanded_data, dummy_good_bands)
    assert len(cleaned_data.columns) == len(dummy_good_bands) + len(preprocessing.LABEL_COLS) # extra columns for coordinates, image name, and pixel class
    assert len(cleaned_data) == len(expanded_data) 
    assert all([str(band) in cleaned_data.columns for band in dummy_good_bands])
    assert all([str(band) not in cleaned_data.columns for band in [0, 1]])

def test_detect_bad_values(expanded_data):
    # sample 0 val 0 should be replaced by 0.2 from same image, same class
    # samples 0 and 1 val 9 should be replaced in class with 0.2, rather than 0.9
    # sample 4 val 9 should be replaced by 0.2 from same image, different class
    # sample 4 val 6 should be replaced by 0.1 from same image, different class
    # sample 5 val 1 should be replaced by 0.1 from same image, same class
    # sample 5 val 3 should be replaced by 0.1 from same image, different class
    # sample 8 val 3 should be replaced by 0.1 from same image, different class
    # sample 9 val 2 should be replaced by 0.2 from different images.
    # sample_spectrum_0 = [1.5, 0.2, 0.2, 0.2, 0.1, 0.1, 0.3, 0.3, 0.1, 1.5]
    # sample_spectrum_1 = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.3, 0.3, 0.1, 1.5]
    # sample_spectrum_2 = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.3, 0.3, 0.1, 0.2]
    # sample_spectrum_3 = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.3, 0.3, 0.1, 0.2]
    # sample_spectrum_4 = [0.2, 0.2, 0.2, 0.2, 0.1, 1.5, 0.3, 0.3, 0.1, 0.9]
    # sample_spectrum_5 = [1.5, 0.2, 1.5, 0.2, 0.1, 0.1, 0.3, 0.3, 0.1, 0.2] 
    # sample_spectrum_6 = [0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.3, 0.3, 0.1, 0.2]
    # sample_spectrum_7 = [0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.3, 0.3, 0.1, 0.2]
    # sample_spectrum_8 = [0.1, 0.2, 1.5, 0.2, 0.1, 0.1, 0.3, 0.3, 0.1, 0.2]
    # sample_spectrum_9 = [0.1, 1.5, 0.1, 0.2, 0.1, 0.1, 0.3, 0.3, 0.1, 0.2]

    assert preprocessing.detect_bad_values(expanded_data.iloc[0].to_frame().T, threshold=1) == True
    assert preprocessing.detect_bad_values(expanded_data.iloc[6].to_frame().T, threshold=1) == False
    assert preprocessing.detect_bad_values(expanded_data.iloc[0].to_frame().T, threshold=2) == False
    assert preprocessing.detect_bad_values(expanded_data, threshold=1) == True
    assert preprocessing.detect_bad_values(expanded_data.iloc[6:8], threshold=1) == False

def test_impute_column_mean(expanded_data):
    column_names = expanded_data.columns
    imputed_data = preprocessing.impute_column_mean(expanded_data, threshold=1)
    col_name_set = set(column_names+imputed_data.columns) # Ensure no columns were accidentally dropped
    assert len(col_name_set) == len(column_names)
    assert len(imputed_data) == len(expanded_data)
    assert round(imputed_data.iloc[9]["1"],3) == 0.2
    assert round(imputed_data.iloc[6]["2"],3) == 0.1
    assert round(imputed_data.iloc[0]["0"],3) == 0.15
    imputed_data = preprocessing.impute_column_mean(expanded_data, threshold=0.8)
    assert round(imputed_data.iloc[4]["9"],3) == 0.2

@pytest.mark.filterwarnings('ignore::FutureWarning')
def test_impute_bad_values(expanded_data):
    column_names = expanded_data.columns
    imputed_data = preprocessing.impute_bad_values(expanded_data, threshold=1)
    col_name_set = set(column_names+imputed_data.columns) # Ensure no columns were accidentally dropped
    assert len(col_name_set) == len(column_names)
    assert len(imputed_data) == len(expanded_data)
    assert round(imputed_data.iloc[0]["0"],3) == 0.2 # Same image, same class
    assert round(imputed_data.iloc[1]["9"],3) == 0.2 # Same image, same class
    assert round(imputed_data.iloc[4]["5"],3) == 0.1 # Same image, different class
    assert round(imputed_data.iloc[5]["0"],3) == 0.1 # Same image, same class
    assert round(imputed_data.iloc[5]["2"],3) == 0.1 # Same image, different class
    assert round(imputed_data.iloc[8]["2"],3) == 0.1 # Same image, different class
    assert round(imputed_data.iloc[9]["1"],3) == 0.2 # Different image, different class



    


