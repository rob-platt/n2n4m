from pandas.core.frame import DataFrame
import pytest
import pandas as pd
import numpy as np

from n2n4m.wavelengths import ALL_WAVELENGTHS, PLEBANI_WAVELENGTHS
import n2n4m.preprocessing as preprocessing


@pytest.fixture
def raw_data():
    # Small sample dataset with columns [Coordinates(list), Image_Name(str), Pixel_Class, Spectrum(list)].
    data = pd.read_json(
        "tests/test_preprocessing/sample_data_v2.json", dtype={"Image_Name": "string"}
    )
    return data


def test_load_dataset(raw_data: DataFrame):
    sample_data_load = preprocessing.load_dataset(
        "tests/test_preprocessing/sample_data_v2.json"
    )
    assert type(sample_data_load) == pd.DataFrame
    assert sample_data_load.equals(raw_data)
    pytest.raises(
        FileNotFoundError,
        preprocessing.load_dataset,
        "tests/test_preprocessing/not_a_file.json",
    )
    pytest.raises(
        ValueError,
        preprocessing.load_dataset,
        "tests/test_preprocessing/test_preprocessing.py",
    )


@pytest.fixture
def expanded_data():
    # Same sample as raw_data, with spectrum column expanded to 1 column per band.
    data = pd.read_json(
        "tests/test_preprocessing/sample_data_v2_expanded.json",
        dtype={"Image_Name": "string"},
    )
    return data


def test_expand_spectrum(raw_data: DataFrame, expanded_data: DataFrame):
    bands = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    sample_data_expanded = preprocessing.expand_dataset(raw_data, bands=bands)
    assert type(sample_data_expanded) == pd.DataFrame
    assert set(sample_data_expanded.columns) == set(expanded_data.columns)
    assert len(sample_data_expanded) == len(expanded_data)


def test_drop_bad_bands(expanded_data: DataFrame):
    dummy_good_bands = (
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    )  # bands 0 and 1 are bad in dummy example
    dummy_bad_bands = [0, 1]
    cleaned_data = preprocessing.drop_bad_bands(expanded_data, dummy_good_bands)
    assert type(cleaned_data) == pd.DataFrame
    assert len(cleaned_data.columns) == len(dummy_good_bands) + len(
        preprocessing.LABEL_COLS
    )  # extra columns for coordinates, image name, and pixel class
    assert len(cleaned_data) == len(expanded_data)  # Check no rows removed
    assert all(
        [str(band) in cleaned_data.columns for band in dummy_good_bands]
    )  # Check that all good bands present
    assert all(
        [str(band) not in cleaned_data.columns for band in dummy_bad_bands]
    )  # Check all bad bands removed


def test_detect_bad_values(expanded_data: DataFrame):
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

    assert (
        preprocessing.detect_bad_values(expanded_data.iloc[0].to_frame().T, threshold=1)
        == True
    )
    assert (
        preprocessing.detect_bad_values(expanded_data.iloc[6].to_frame().T, threshold=1)
        == False
    )
    assert (
        preprocessing.detect_bad_values(expanded_data.iloc[0].to_frame().T, threshold=2)
        == False
    )
    assert preprocessing.detect_bad_values(expanded_data, threshold=1) == True
    assert (
        preprocessing.detect_bad_values(expanded_data.iloc[6:8], threshold=1) == False
    )


def test_impute_column_mean(expanded_data: DataFrame):
    column_names = expanded_data.columns
    imputed_data = preprocessing.impute_column_mean(expanded_data, threshold=1)
    col_name_set = set(
        column_names + imputed_data.columns
    )  # Ensure no columns were accidentally dropped
    assert type(imputed_data) == pd.DataFrame
    assert len(col_name_set) == len(column_names)
    assert len(imputed_data) == len(expanded_data)
    assert np.isclose(imputed_data.iloc[9]["1"], 0.2)
    assert np.isclose(imputed_data.iloc[6]["2"], 0.1)
    assert np.isclose(imputed_data.iloc[0]["0"], 0.15)
    imputed_data = preprocessing.impute_column_mean(expanded_data, threshold=0.8)
    assert np.isclose(imputed_data.iloc[4]["9"], 0.2)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_impute_bad_values(expanded_data: DataFrame):
    column_names = expanded_data.columns
    imputed_data = preprocessing.impute_bad_values(expanded_data, threshold=1)
    col_name_set = set(
        column_names + imputed_data.columns
    )  # Ensure no columns were accidentally dropped
    assert type(imputed_data) == pd.DataFrame
    assert len(col_name_set) == len(column_names)
    assert len(imputed_data) == len(expanded_data)
    assert np.isclose(imputed_data.iloc[0]["0"], 0.2)  # Same image, same class
    assert np.isclose(imputed_data.iloc[1]["9"], 0.2)  # Same image, same class
    assert np.isclose(imputed_data.iloc[4]["5"], 0.1)  # Same image, different class
    assert np.isclose(imputed_data.iloc[5]["0"], 0.1)  # Same image, same class
    assert np.isclose(imputed_data.iloc[5]["2"], 0.1)  # Same image, different class
    assert np.isclose(imputed_data.iloc[8]["2"], 0.1)  # Same image, different class
    assert np.isclose(
        imputed_data.iloc[9]["1"], 0.2
    )  # Different image, different class


def test_get_linear_interp_spectra(expanded_data: DataFrame):
    bands = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    sample_spectra = expanded_data.iloc[8, 3:].values.astype(
        float
    )  # Single spectra, only the bands, no labels
    linear_interp = preprocessing.get_linear_interp_spectra(
        sample_spectra, lower_bound=0, upper_bound=5, wavelengths=bands
    )
    assert type(linear_interp) == np.ndarray
    assert len(linear_interp) == 5  # 5 bands between 0 and 5
    assert round(linear_interp[0], 3) == 0.42


def test_detect_artefact(expanded_data: DataFrame):
    bands = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    sample_bad_spectra = expanded_data.iloc[8, 3:].values.astype(
        float
    )  # Single spectra, only the bands, no labels
    sample_good_spectra = expanded_data.iloc[1, 3:].values.astype(float)
    assert (
        preprocessing.detect_artefact(
            sample_bad_spectra,
            lower_bound=0,
            upper_bound=6,
            wavelengths=bands,
            threshold=0.5,
        )
        == True
    )
    assert (
        preprocessing.detect_artefact(
            sample_good_spectra,
            lower_bound=0,
            upper_bound=6,
            wavelengths=bands,
            threshold=0.5,
        )
        == False
    )
    assert (
        preprocessing.detect_artefact(
            sample_bad_spectra,
            lower_bound=0,
            upper_bound=6,
            wavelengths=bands,
            threshold=0.6,
        )
        == True
    )  # Change boundary threshold


def test_impute_artefacts(expanded_data: DataFrame):
    bands = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    imputed_data = preprocessing.impute_artefacts(
        expanded_data, lower_bound=0, upper_bound=6, wavelengths=bands, threshold=0.5
    )
    assert type(imputed_data) == pd.DataFrame
    assert imputed_data.shape == expanded_data.shape


def test_generate_noisy_pixels(expanded_data: DataFrame):
    spectra = expanded_data.drop(columns=preprocessing.LABEL_COLS)
    noisy_data = preprocessing.generate_noisy_pixels(spectra)
    assert type(noisy_data) == pd.DataFrame
    assert noisy_data.shape == spectra.shape


def test_train_test_split(expanded_data: DataFrame):
    train, test = preprocessing.train_test_split(expanded_data)
    assert type(train) == pd.DataFrame
    assert type(test) == pd.DataFrame
    assert len(train) + len(test) == len(expanded_data)
    assert len(train.columns) == len(test.columns) == len(expanded_data.columns)
    assert "0BABA" in train["Image_Name"].values
    assert "0BABA" not in test["Image_Name"].values
    assert "0A053" in test["Image_Name"].values
    assert "0A053" not in train["Image_Name"].values


def test_train_validation_split(expanded_data: DataFrame):
    train, validation = preprocessing.train_validation_split(expanded_data)
    assert type(train) == pd.DataFrame
    assert type(validation) == pd.DataFrame
    assert len(train) + len(validation) == len(expanded_data)
    assert len(train.columns) == len(validation.columns) == len(expanded_data.columns)
    assert "0BABA" in train["Image_Name"].values
    assert "0BABA" not in validation["Image_Name"].values
    assert "093BE" in validation["Image_Name"].values
    assert "093BE" not in train["Image_Name"].values


@pytest.fixture
def noisy_expanded_data():
    # Same sample as expanded_data, just bands >=5 relabelled as noisy. Actual spectra are unchanged.
    data = pd.read_json(
        "tests/test_preprocessing/sample_data_v2_expanded_noisy.json",
        dtype={"Image_Name": "string"},
    )
    return data


def test_split_features_targets_anciliary(noisy_expanded_data: DataFrame):
    features, targets, anciliary = preprocessing.split_features_targets_anciliary(
        noisy_expanded_data
    )
    assert type(features) == pd.DataFrame
    assert type(targets) == pd.DataFrame
    assert type(anciliary) == pd.DataFrame
    assert len(features.columns) == 5
    assert len(targets.columns) == 5
    assert set(anciliary.columns) == set(preprocessing.LABEL_COLS)
    assert len(features) == len(targets) == len(anciliary) == len(noisy_expanded_data)
