import pytest


import n2n4m.summary_parameters as summary_parameters


def test_wavelength_weights():
    a, b = summary_parameters.wavelength_weights(1, 2, 3)
    assert a == 0.5
    assert b == 0.5
    a, b = summary_parameters.wavelength_weights(1, 1, 2)
    assert a == 1
    assert b == 0