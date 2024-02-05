import pytest
import numpy as np


import n2n4m.summary_parameters as summary_parameters


def test_wavelength_weights():
    a, b = summary_parameters.wavelength_weights(1, 2, 3)
    assert a == 0.5
    assert b == 0.5
    a, b = summary_parameters.wavelength_weights(1, 1, 2)
    assert a == 1
    assert b == 0


def test_interpolated_center_wavelength_reflectance():
    wavelengths = (1, 2, 3)
    short_ref = np.array([1])
    long_ref = np.array([3])
    interpolated_reflectance = summary_parameters.interpolated_center_wavelength_reflectance(short_ref, wavelengths, long_ref)
    assert type(interpolated_reflectance) == np.ndarray
    assert interpolated_reflectance == 2


def test_band_depth_calculation():
    spectrum = np.array([[2, 1, 2]])
    all_wavelengths = (1, 2, 3)
    bd_wavelengths = (1, 2, 3)
    kernel_sizes = (1, 1, 1)
    band_depth = summary_parameters.band_depth_calculation(spectrum, all_wavelengths, bd_wavelengths, kernel_sizes)
    assert type(band_depth) == np.ndarray
    assert band_depth == 0.5
