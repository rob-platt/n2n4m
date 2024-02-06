import pytest
import numpy as np


import n2n4m.cotcat_denoise as cotcat_denoise


def test_sharpening_median_filter():
    # 1x1x12 (1 row, 1 col, 12 bands), single anomalous value, median value of 1
    spectrum = np.array([[[1, 1, 1, 1, 1, 1.5, 1, 1, 1, 1, 1, 1]]]) 
    correct_filtered_spectrum = np.array([[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]])
    filtered_spectrum = cotcat_denoise.sharpening_median_filter(spectrum)
    assert type(filtered_spectrum) == np.ndarray
    assert filtered_spectrum.shape == spectrum.shape # Shape should be retained
    assert np.allclose(filtered_spectrum, correct_filtered_spectrum)
    # 2x2x12 (2 row, 2 col, 12 bands), single anomalous value, median value of 1.5
    spectra = np.array([[spectrum.squeeze(), spectrum.squeeze()], [spectrum.squeeze(), spectrum.squeeze()]])
    filtered_spectra = cotcat_denoise.sharpening_median_filter(spectra)
    assert type(filtered_spectra) == np.ndarray
    assert filtered_spectra.shape == spectra.shape # Shape should be retained
    # Loop through all filtered spectra and check if they are all equal to the correct filtered spectrum
    assert all([np.allclose(filt_spec, correct_filtered_spectrum) for filt_spec in filtered_spectra.reshape(4, 12)]) == True


def test_moving_median_filter():
    spectrum = np.array([1, 1, 1, 2, 1, 2, 1])
    # Should remove the 2 in the centre, but retain the value idx < FILTER_SIZE//2 from the end
    correct_filtered_spectrum = np.array([1, 1, 1, 1, 1, 2, 1]) 
    spectra = np.array([[spectrum, spectrum], [spectrum, spectrum]])
    filtered_spectra = cotcat_denoise.moving_median_filter(spectra)
    assert type(filtered_spectra) == np.ndarray
    assert filtered_spectra.shape == spectra.shape # Shape should be retained
    # Loop through all filtered spectra and check if they are all equal to the correct filtered spectrum
    assert all([np.allclose(filt_spec, correct_filtered_spectrum) for filt_spec in filtered_spectra.reshape(4, 7)]) == True


def test_moving_mean_filter():
    spectrum = np.array([1, 1, 1, 2, 1, 2, 1], dtype=np.float32)
    correct_filtered_spectrum = np.array([1, 1, 1.2, 1.4, 1.4, 2, 1], dtype=np.float32)
    spectra = np.array([[spectrum, spectrum], [spectrum, spectrum]])
    filtered_spectra = cotcat_denoise.moving_mean_filter(spectra)
    assert type(filtered_spectra) == np.ndarray
    assert filtered_spectra.shape == spectra.shape # Shape should be retained
    # Loop through all filtered spectra and check if they are all equal to the correct filtered spectrum
    assert all([np.allclose(filt_spec, correct_filtered_spectrum) for filt_spec in filtered_spectra.reshape(4, 7)]) == True


def test_cotcat_denoise():
    # 1x1x12 (1 row, 1 col, 12 bands), single anomalous value, median value of 1
    spectrum = np.ones((1, 1, 50))
    spectrum[0, 0, 10] = 1.5
    correct_filtered_spectrum = np.ones((1, 1, 50))
    wavelengths_no_break = tuple(np.arange(0.001, 0.051, 0.001))
    filtered_spectrum = cotcat_denoise.cotcat_denoise(spectrum, wavelengths_no_break)
    assert type(filtered_spectrum) == np.ndarray
    assert filtered_spectrum.shape == spectrum.shape # Shape should be retained
    assert np.allclose(filtered_spectrum, correct_filtered_spectrum)

    # Now add a gap in the wavelengths, should treat each section separately but still return the same shape
    wavelengths_with_break = tuple(np.concatenate([np.arange(0.001, 0.026, 0.001), np.arange(0.045, 0.071, 0.001)]))
    filtered_spectrum = cotcat_denoise.cotcat_denoise(spectrum, wavelengths_with_break)
    assert type(filtered_spectrum) == np.ndarray
    assert filtered_spectrum.shape == spectrum.shape # Shape should be retained
    assert np.allclose(filtered_spectrum, correct_filtered_spectrum)

    # Now add a gap in wavelengths that is too close to the start/end, should raise a ValueError
    wavelengths_with_bad_break = tuple(np.concatenate([np.arange(0.001, 0.003, 0.001), np.arange(0.050, 0.098, 0.001)]))
    pytest.raises(ValueError, cotcat_denoise.cotcat_denoise, spectrum, wavelengths_with_bad_break)