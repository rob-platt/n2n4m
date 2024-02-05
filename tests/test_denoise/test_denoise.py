import pytest
import numpy as np

import n2n4m.denoise as denoise
from n2n4m.wavelengths import ALL_WAVELENGTHS


def test_band_index_mask():
    bands_to_keep = (1.001350, 1.007900, 1.014450, 1.021000, 1.027550) # First 5 bands in ALL_WAVELENGTHS
    include_bands_indices, exclude_bands_indices = denoise.band_index_mask(bands_to_keep)
    assert type(include_bands_indices) == np.ndarray and type(exclude_bands_indices) == np.ndarray
    assert len(include_bands_indices) + len(exclude_bands_indices) == len(ALL_WAVELENGTHS) # All bands accounted for
    assert tuple(include_bands_indices) == (0, 1, 2, 3, 4)  # First 5 bands
    assert set(include_bands_indices).isdisjoint(exclude_bands_indices)  # No overlap between include and exclude bands

    