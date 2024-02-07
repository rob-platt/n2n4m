import pytest
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.exceptions import NotFittedError
import importlib.resources

import n2n4m.n2n4m_denoise as n2n4m_denoise
from n2n4m.wavelengths import ALL_WAVELENGTHS
from n2n4m.model import Noise2Noise1D


TRAINED_MODEL_FILEPATH = importlib.resources.files("n2n4m") / "data/trained_model_weights.pt"
FITTED_SCALER_FILEPATH = importlib.resources.files("n2n4m") / "data/input_standardiser.pkl"

def test_band_index_mask():
    bands_to_keep = (
        1.001350,
        1.007900,
        1.014450,
        1.021000,
        1.027550,
    )  # First 5 bands in ALL_WAVELENGTHS
    include_bands_indices, exclude_bands_indices = n2n4m_denoise.band_index_mask(
        bands_to_keep
    )
    assert (
        type(include_bands_indices) == np.ndarray
        and type(exclude_bands_indices) == np.ndarray
    )
    assert len(include_bands_indices) + len(exclude_bands_indices) == len(
        ALL_WAVELENGTHS
    )  # All bands accounted for
    assert tuple(include_bands_indices) == (0, 1, 2, 3, 4)  # First 5 bands
    assert set(include_bands_indices).isdisjoint(
        exclude_bands_indices
    )  # No overlap between include and exclude bands


def test_clip_bands():
    spectra = np.random.rand(1, len(ALL_WAVELENGTHS))  # Random spectra
    bands_to_keep = (
        1.001350,
        1.007900,
        1.014450,
        1.021000,
        1.027550,
    )  # First 5 bands in ALL_WAVELENGTHS
    interior_bands, exterior_bands = n2n4m_denoise.clip_bands(spectra, bands_to_keep)
    assert type(interior_bands) == np.ndarray and type(exterior_bands) == np.ndarray
    assert (
        interior_bands.shape[0] == spectra.shape[0]
        and exterior_bands.shape[0] == spectra.shape[0]
    )
    assert (
        interior_bands.shape[1] + exterior_bands.shape[1] == spectra.shape[1]
    )  # All bands accounted for
    assert interior_bands.shape[1] == len(bands_to_keep)  # Only the bands to keep
    assert set(interior_bands[0]) == set(spectra[0][0:5])  # The first 5 bands


def test_combine_bands():
    bands_to_keep = (
        1.001350,
        1.007900,
        1.014450,
        1.021000,
        1.027550,
    )  # First 5 bands in ALL_WAVELENGTHS
    clipped_data = np.random.rand(
        1, len(bands_to_keep)
    )  # Random spectra with only the bands to keep
    extra_data = np.random.rand(
        1, len(ALL_WAVELENGTHS) - len(bands_to_keep)
    )  # Random spectra with the bands to exclude
    combined_data = n2n4m_denoise.combine_bands(clipped_data, extra_data, bands_to_keep)
    assert type(combined_data) == np.ndarray
    assert combined_data.shape[1] == len(ALL_WAVELENGTHS)  # All bands accounted for
    assert (
        combined_data[0, :5].all() == clipped_data[0].all()
    )  # Ensure the clipped data was put in the correct place (first 5 bands)
    assert (
        combined_data.shape[0] == clipped_data.shape[0] == extra_data.shape[0]
    )  # Ensure the number of samples is the same


def test_load_scaler():
    untrained_scaler_filepath = "tests/test_n2n4m_denoise/untrained_scaler.pkl"
    scaler = n2n4m_denoise.load_scaler(FITTED_SCALER_FILEPATH)
    assert type(scaler) == type(RobustScaler())
    pytest.raises(
        NotFittedError, n2n4m_denoise.load_scaler, untrained_scaler_filepath
    )  # Ensure the scaler is fitted


def test_instantiate_default_model():
    model = n2n4m_denoise.instantiate_default_model(TRAINED_MODEL_FILEPATH)
    assert type(model) == Noise2Noise1D
    assert model.training == False  # Model should be in evaluation mode
