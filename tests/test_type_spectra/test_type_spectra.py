import pytest
import pandas as pd
import numpy as np

import n2n4m.type_spectra as type_spectra


def test_get_mineral_class():
    sample = pd.Series({"Pixel_Class": [10]})
    assert type_spectra.get_mineral_class(sample) == "Serpentine"


def test_get_type_spectra_class():
    sample = pd.Series({"Pixel_Class": [10]})
    assert type_spectra.get_type_spectra_class(sample) == "serpentine"


def test_get_type_spectra_name():
    sample = pd.Series({"Pixel_Class": [10]})
    assert type_spectra.get_type_spectra_name(sample) == "Serpentine"


def test_read_type_spectra():
    good_filepath = "tests/test_type_spectra/crism_typespec_serpentine.tab"
    bad_filepath = "tests/test_type_spectra/crism_typespec_bastantite.tab"

    spectrum = type_spectra.read_type_spectra(good_filepath)

    assert type(spectrum) == pd.Series
    assert len(spectrum) == 10
    assert type(spectrum.index[0]) == str
    assert type(spectrum.values[0]) == np.float64
    pytest.raises(FileNotFoundError, type_spectra.read_type_spectra, bad_filepath)


def test_get_type_spectra():
    good_dir = "tests/test_type_spectra"
    bad_dir = "tests/bad_dir"
    good_sample = pd.Series({"Pixel_Class": [10]})
    bad_sample = pd.DataFrame({"Pixel_Class": [10]})

    spectrum = type_spectra.get_type_spectra(good_sample, good_dir)

    pytest.raises(IOError, type_spectra.get_type_spectra, good_sample, bad_dir)
    pytest.raises(TypeError, type_spectra.get_type_spectra, bad_sample, good_dir)


def test_clip_type_spectra():
    good_dir = "tests/test_type_spectra"
    mineral_sample = pd.Series({"Pixel_Class": [10], "1.38175": 1, "1.38832": 2})

    spectrum = type_spectra.get_type_spectra(mineral_sample, good_dir)
    spectrum = type_spectra.clip_type_spectra(mineral_sample, spectrum)

    assert type(spectrum) == pd.Series
    assert len(spectrum) == 2
    assert spectrum.index[0] == "1.38175"
    assert spectrum.index[1] == "1.38832"
