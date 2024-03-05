import numpy as np

# If implementing new summary parameters, add them to the IMPLEMENTED_SUMMARY_PARAMETERS dictionary below for them to show in CRISMImage.summary_parameters.


def _wavelength_weights(
    short_wavelength: float,
    center_wavelength: float,
    long_wavelength: float,
) -> tuple[float, float]:
    """
    Calculate the wavelength weights for a given wavelength range.

    Parameters
    ----------
    short_wavelength : float
        Short wavelength.
    center_wavelength : float
        Center wavelength.
    long_wavelength : float
        Long wavelength.

    Returns
    -------
    a : float
        Weight for short wavelength.
    b : float
        Weight for long wavelength.
    """
    b = (center_wavelength - short_wavelength) / (long_wavelength - short_wavelength)
    a = 1 - b
    return a, b


def _interpolated_center_wavelength_reflectance(
    short_ref: np.ndarray,
    bd_wavelengths: tuple[float, float, float],
    long_ref: np.ndarray,
) -> np.ndarray:
    """
    Calculate the centre wavelength reflectance of spectra.
    Weighted average of short and long wavelength reflectance values, i.e. linear interpolation.
    Weight determined by distance from centre wavelength.

    Parameters
    ----------
    short_ref : np.ndarray
        Median reflectance of kernel centred at short wavelength.
        Shape (n_spectra,)
    bd_wavelengths: tuple[float, float, float]
        Wavelengths to calculate the centre wavelength reflectance for.
        (short_wavelength, center_wavelength, long_wavelength)
    long_ref : np.ndarray
        Median reflectance of kernel centred at long wavelength.
        Shape (n_spectra,)

    Returns
    -------
    interpolated_center_ref : np.ndarray
        Interpolated centre wavelength reflectance.
        Shape (n_spectra,)
    """
    a, b = _wavelength_weights(bd_wavelengths[0], bd_wavelengths[1], bd_wavelengths[2])
    return a * short_ref + b * long_ref


def _band_depth_calculation(
    spectra: np.ndarray,
    all_wavelengths: tuple[float, ...],
    bd_wavelengths: tuple[float, float, float],
    kernel_sizes: tuple[int, int, int],
) -> np.ndarray:
    """
    Calculate the band depth for a given set of wavelengths.

    Parameters
    ----------
    spectra : np.ndarray
        Spectra to calculate band depth for.
        Shape (n_spectra, n_wavelengths)
    all_wavelengths : tuple
        Wavelengths corresponding to spectra.
        Shape (n_wavelengths,)
    bd_wavelengths : tuple
        Wavelengths to calculate band depth for.
        (short_wavelength, center_wavelength, long_wavelength)
    kernel_sizes : tuple
        Kernel sizes to use for each wavelength.
        (short_wavelength, center_wavelength, long_wavelength)

    Returns
    -------
    band_depth : np.ndarray
        Band depth values for each spectra.
        Shape (n_spectra,)
    """
    short_wavelength = bd_wavelengths[0]
    center_wavelength = bd_wavelengths[1]
    long_wavelength = bd_wavelengths[2]

    short_ref = np.zeros(spectra.shape[0])
    center_ref = np.zeros(spectra.shape[0])
    long_ref = np.zeros(spectra.shape[0])

    half_kernel_sizes = [kernel_size // 2 for kernel_size in kernel_sizes]

    short_ref = np.median(
        spectra[
            :,
            all_wavelengths.index(short_wavelength)
            - half_kernel_sizes[0] : all_wavelengths.index(short_wavelength)
            + half_kernel_sizes[0]
            + 1,
        ],
        axis=1,
    )
    center_ref = np.median(
        spectra[
            :,
            all_wavelengths.index(center_wavelength)
            - half_kernel_sizes[1] : all_wavelengths.index(center_wavelength)
            + half_kernel_sizes[1]
            + 1,
        ],
        axis=1,
    )
    long_ref = np.median(
        spectra[
            :,
            all_wavelengths.index(long_wavelength)
            - half_kernel_sizes[2] : all_wavelengths.index(long_wavelength)
            + half_kernel_sizes[2]
            + 1,
        ],
        axis=1,
    )

    interpolated_center_ref = _interpolated_center_wavelength_reflectance(
        short_ref, bd_wavelengths, long_ref
    )
    band_depth = center_ref / interpolated_center_ref
    return band_depth


def hyd_femg_clay_index_calculation(
    spectra: np.ndarray,
    wavelengths: tuple[float, ...],
) -> np.ndarray:
    """
    Calculate the summary parameter Hydrated Fe/Mg Clay Index across an image.
    Index from [1].

    Parameters
    ----------
    spectra : np.ndarray
        Spectra to calculate Hydrated Fe/Mg Clay Index for.
        Shape (n_spectra, n_wavelengths)
    wavelengths : tuple[float, ...]
        Wavelengths corresponding to spectra.
        Shape (n_wavelengths,)

    Returns
    -------
    bd1750 : np.ndarray
        Hydrated Fe/Mg Clay Index values for each spectra.
        Shape (n_spectra,)

    References
    ----------
    1. Loizeau D, Quantin-Nataf C, Carter J, Flahaut J, Thollot P, Lozac'h L, et al.
    Quantifying widespread aqueous surface weathering on Mars: The plateaus south of Coprates Chasma.
    Icarus. 2018 Mar 1;302:451-69.

    """

    femg_clays = np.zeros(spectra.shape[0])

    bd14 = 1 - _band_depth_calculation(
        spectra, wavelengths, (1.33578, 1.41459, 1.55264), (5, 3, 5)
    )
    bd14[bd14 < 0] = 0
    bd19 = 1 - _band_depth_calculation(
        spectra, wavelengths, (1.86212, 1.92146, 2.07985), (5, 5, 5)
    )
    bd19[bd19 < 0] = 0
    bd229 = 1 - _band_depth_calculation(
        spectra, wavelengths, (2.15251, 2.28472, 2.34426), (3, 5, 3)
    )
    bd229[bd229 < 0] = 0
    bd231 = 1 - _band_depth_calculation(
        spectra, wavelengths, (2.15251, 2.30456, 2.34426), (3, 3, 3)
    )
    bd231[bd231 < 0] = 0
    bd232 = 1 - _band_depth_calculation(
        spectra, wavelengths, (2.15251, 2.32441, 2.34426), (3, 3, 3)
    )
    bd232[bd232 < 0] = 0
    bd238 = 1 - _band_depth_calculation(
        spectra, wavelengths, (2.34426, 2.38396, 2.4303), (3, 3, 3)
    )
    bd238[bd238 < 0] = 0
    bd240 = 1 - _band_depth_calculation(
        spectra, wavelengths, (2.34426, 2.3972, 2.4303), (3, 3, 3)
    )
    bd240[bd240 < 0] = 0

    femg_clays = bd14 + bd19 + bd229 + bd231 + bd232 + bd238 + bd240

    return femg_clays


def d2300_calculation(
    spectra: np.ndarray,
    wavelengths: tuple[float, ...],
) -> np.ndarray:
    """
    Calculate the dropoff at 2300nm across an image.
    Highlights Mg,Fe-OH minerals, as well as Mg-Carbonates, and CO2 ice [1].

    Parameters
    ----------
    spectra : np.ndarray
        Spectra to calculate dropoff at 2300nm for.
        Shape (n_spectra, n_wavelengths)
    wavelengths : tuple[float, ...]
        Wavelengths corresponding to spectra.
        Shape (n_wavelengths,)

    Returns
    -------
    d2300 : np.ndarray
        Dropoff at 2300nm values for each spectra.
        Shape (n_spectra,)

    References
    ----------
    1. Viviano-Beck CE, Seelos FP, Murchie SL, Kahn EG, Seelos KD, Taylor HW, et al.
    Revised CRISM spectral parameters and summary products based on the currently
    detected mineral diversity on Mars.
    Journal of Geophysical Research: Planets. 2014;119(6):1403-31.
    """
    d2300 = np.zeros(spectra.shape[0])

    bd2290 = _band_depth_calculation(
        spectra, wavelengths, (1.81598, 2.30456, 2.52951), (5, 3, 5)
    )
    bd2320 = _band_depth_calculation(
        spectra, wavelengths, (1.81598, 2.32441, 2.52951), (5, 3, 5)
    )
    bd2330 = _band_depth_calculation(
        spectra, wavelengths, (1.81598, 2.23182, 2.52951), (5, 3, 5)
    )
    bd2120 = _band_depth_calculation(
        spectra, wavelengths, (1.81598, 2.11948, 2.52951), (5, 5, 5)
    )
    bd2170 = _band_depth_calculation(
        spectra, wavelengths, (1.81598, 2.17233, 2.52951), (5, 5, 5)
    )
    bd2210 = _band_depth_calculation(
        spectra, wavelengths, (1.81598, 2.21199, 2.52951), (5, 5, 5)
    )

    d2300 = 1 - ((bd2290 + bd2320 + bd2330) / (bd2120 + bd2170 + bd2210))

    return d2300


def bd1750_calculation(
    spectra: np.ndarray,
    wavelengths: tuple[float, ...],
) -> np.ndarray:
    """
    Calculate the summary parameter BD1750 across an image.
    BD1750 used to identify presence of absorption feature at 1.75um, present in Alunite and Gypsum [1].
    Negative values are clipped to 0.

    Parameters
    ----------
    spectra : np.ndarray
        Spectra to calculate BD1750 for.
        Shape (n_spectra, n_wavelengths)
    wavelengths : tuple[float, ...]
        Wavelengths corresponding to spectra.
        Shape (n_wavelengths,)

    Returns
    -------
    bd1750 : np.ndarray
        BD1750 values for each spectra.
        Shape (n_spectra,)

    References
    ----------
    1. Viviano-Beck CE, Seelos FP, Murchie SL, Kahn EG, Seelos KD, Taylor HW, et al.
    Revised CRISM spectral parameters and summary products based on the currently
    detected mineral diversity on Mars.
    Journal of Geophysical Research: Planets. 2014;119(6):1403-31.
    """
    bd1750 = np.zeros(spectra.shape[0])

    bd1750 = 1 - _band_depth_calculation(
        spectra, wavelengths, (1.55264, 1.75009, 1.81598), (1, 1, 1)
    )
    bd1750[bd1750 < 0] = 0

    return bd1750


def bd175_calculation(
    spectra: np.ndarray,
    wavelengths: tuple[float, ...],
) -> np.ndarray:
    """Calculate hte BD175 summary parameter across an image.
    BD175 used to identify presence of absorption feature at 1.75um, present in Alunite and Gypsum [1].
    Negative values are clipped to 0.

    Parameters
    ----------
    spectra : np.ndarray
        Spectra to calculate BD175 for.
        Shape (n_spectra, n_wavelengths)
    wavelengths : tuple[float, ...]
        Wavelengths corresponding to spectra.
        Shape (n_wavelengths,)

    Returns
    -------
    bd175 : np.ndarray
        BD175 values for each spectra.
        Shape (n_spectra,)

    References
    ----------
    1. Bultel B, Quantin C, Lozac'h L. Description of CoTCAT (Complement to CRISM Analysis Toolkit).
    IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing.
    2015 Jun;8(6):3039-49.
    """

    bd175 = np.ones(spectra.shape[0])

    bd175 = 1 - _band_depth_calculation(
        spectra, wavelengths, (1.55264, 1.75009, 1.81598), (1, 1, 1)
    )
    bd175[bd175 < 0] = 0

    return bd175


IMPLEMENTED_SUMMARY_PARAMETERS = {
    "hyd_femg_clay_index": hyd_femg_clay_index_calculation,
    "d2300": d2300_calculation,
    "bd1750": bd1750_calculation,
    "bd175": bd175_calculation,
}
