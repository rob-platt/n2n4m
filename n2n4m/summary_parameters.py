import numpy as np

# If implementing new summary parameters,
# add them to the IMPLEMENTED_SUMMARY_PARAMETERS dictionary below for them
# to show in CRISMImage.summary_parameters.


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
    b = (center_wavelength - short_wavelength) / (
        long_wavelength - short_wavelength
    )
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
    a, b = _wavelength_weights(
        bd_wavelengths[0], bd_wavelengths[1], bd_wavelengths[2]
    )
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


def _relative_band_depth_calculation(
    spectra: np.ndarray,
    all_wavelengths: tuple[float, ...],
    bd_wavelengths: tuple[float, float, float],
    kernel_sizes: tuple[int, int, int],
) -> np.ndarray:
    """
    An alternative method for calculating band depth, which uses the difference
    between the interpolated center wavelength reflectance and the center
    wavelength reflectance as the numerator of the band depth calculation,
    rather than the center wavelength reflectance itself. This is used for
    some summary parameters e.g. OLINDEX3.


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
    relative_band_depth : np.ndarray
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
    relative_band_depth = (
        interpolated_center_ref - center_ref
    ) / interpolated_center_ref

    return relative_band_depth


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
    1. Loizeau D, Quantin-Nataf C, Carter J, Flahaut J, Thollot P,
    Lozac'h L, et al. Quantifying widespread aqueous surface weathering
    on Mars: The plateaus south of Coprates Chasma.
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
    1. Viviano-Beck CE, Seelos FP, Murchie SL, Kahn EG, Seelos KD, Taylor HW,
    et al. Revised CRISM spectral parameters and summary products based on the
    currently detected mineral diversity on Mars.
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
    BD1750 used to identify presence of absorption feature at 1.75um, present
    in Alunite and Gypsum [1].
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
    1. Viviano-Beck CE, Seelos FP, Murchie SL, Kahn EG, Seelos KD, Taylor HW,
    et al. Revised CRISM spectral parameters and summary products based on the
    currently detected mineral diversity on Mars.
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
    """Calculate the BD175 summary parameter across an image.
    BD175 used to identify presence of absorption feature at 1.75um, present
    in Alunite and Gypsum [1].
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
    1. Bultel B, Quantin C, Lozac'h L. Description of CoTCAT (Complement to
    CRISM Analysis Toolkit).
    IEEE Journal of Selected Topics in Applied Earth Observations and
    Remote Sensing.
    2015 Jun;8(6):3039-49.
    """

    bd175 = np.ones(spectra.shape[0])
    lambda_c_1_idx = wavelengths.index(1.75009)
    lambda_c_2_idx = wavelengths.index(1.75668)
    lambda_l_idx = wavelengths.index(1.77644)
    lambda_s_idx = wavelengths.index(1.69082)

    bd175 = bd175 - ((spectra[:, lambda_c_1_idx] + spectra[:, lambda_c_2_idx])/(spectra[:, lambda_s_idx] + spectra[:, lambda_l_idx]))
    bd175[bd175 < 0] = 0

    return bd175


def olindex3_calculation(
    spectra: np.ndarray,
    wavelengths: tuple[float, ...],
) -> np.ndarray:
    """Calculate the OLINDEX3 summary parameter across an image.
    OLINDEX3 used to identify presence of Olivine.
    Negative values are clipped to 0.

    Parameters
    ----------
    spectra : np.ndarray
        Spectra to calculate OLINDEX3 for.
        Shape (n_spectra, n_wavelengths)
    wavelengths : tuple[float, ...]
        Wavelengths corresponding to spectra.
        Shape (n_wavelengths,)

    Returns
    -------
    olindex3 : np.ndarray
        OLINDEX3 values for each spectra.
        Shape (n_spectra,)

    References
    ----------
    1. Viviano-Beck CE, Seelos FP, Murchie SL, Kahn EG, Seelos KD, Taylor HW,
    et al. Revised CRISM spectral parameters and summary products based on the
    currently detected mineral diversity on Mars.
    Journal of Geophysical Research: Planets. 2014;119(6):1403-31.
    """
    rb1080 = _relative_band_depth_calculation(
        spectra, wavelengths, (1.750090, 1.079960, 2.397200), (7, 7, 7)
    )
    rb1152 = _relative_band_depth_calculation(
        spectra, wavelengths, (1.750090, 1.152060, 2.397200), (7, 7, 7)
    )
    rb1210 = _relative_band_depth_calculation(
        spectra, wavelengths, (1.750090, 1.211090, 2.397200), (7, 7, 7)
    )
    rb1250 = _relative_band_depth_calculation(
        spectra, wavelengths, (1.750090, 1.250450, 2.397200), (7, 7, 7)
    )
    rb1263 = _relative_band_depth_calculation(
        spectra, wavelengths, (1.750090, 1.263570, 2.397200), (7, 7, 7)
    )
    rb1276 = _relative_band_depth_calculation(
        spectra, wavelengths, (1.750090, 1.276700, 2.397200), (7, 7, 7)
    )
    rb1330 = _relative_band_depth_calculation(
        spectra, wavelengths, (1.750090, 1.329210, 2.397200), (7, 7, 7)
    )
    rb1368 = _relative_band_depth_calculation(
        spectra, wavelengths, (1.750090, 1.368610, 2.397200), (7, 7, 7)
    )
    rb1395 = _relative_band_depth_calculation(
        spectra, wavelengths, (1.750090, 1.394890, 2.397200), (7, 7, 7)
    )
    rb1427 = _relative_band_depth_calculation(
        spectra, wavelengths, (1.750090, 1.427730, 2.397200), (7, 7, 7)
    )
    rb1470 = _relative_band_depth_calculation(
        spectra, wavelengths, (1.750090, 1.467160, 2.397200), (7, 7, 7)
    )

    olindex3 = (
        rb1080 * 0.03
        + rb1152 * 0.03
        + rb1210 * 0.03
        + rb1250 * 0.03
        + rb1263 * 0.07
        + rb1276 * 0.07
        + rb1330 * 0.12
        + rb1368 * 0.12
        + rb1395 * 0.14
        + rb1427 * 0.18
        + rb1470 * 0.18
    )

    olindex3[olindex3 < 0] = 0
    return olindex3


def lcpindex2_calculation(
    spectra: np.ndarray,
    wavelengths: tuple[float, ...],
) -> np.ndarray:
    """Calculate the LCPINDEX2 summary parameter across an image.
    LCPINDEX2 used to identify presence of low-calcium pyroxenes.
    Negative values are clipped to 0.

    Parameters
    ----------
    spectra : np.ndarray
        Spectra to calculate LCPINDEX2 for.
        Shape (n_spectra, n_wavelengths)
    wavelengths : tuple[float, ...]
        Wavelengths corresponding to spectra.
        Shape (n_wavelengths,)

    Returns
    -------
    lcpindex2 : np.ndarray
        LCPINDEX2 values for each spectra.
        Shape (n_spectra,)

    References
    ----------
    1. Viviano-Beck CE, Seelos FP, Murchie SL, Kahn EG, Seelos KD, Taylor HW,
    et al. Revised CRISM spectral parameters and summary products based on the
    currently detected mineral diversity on Mars.
    Journal of Geophysical Research: Planets. 2014;119(6):1403-31.
    """
    rb1690 = _relative_band_depth_calculation(
        spectra, wavelengths, (1.559210, 1.690820, 2.450170), (7, 7, 7)
    )
    rb1750 = _relative_band_depth_calculation(
        spectra, wavelengths, (1.559210, 1.750090, 2.450170), (7, 7, 7)
    )
    rb1810 = _relative_band_depth_calculation(
        spectra, wavelengths, (1.559210, 1.809390, 2.450170), (7, 7, 7)
    )
    rb1870 = _relative_band_depth_calculation(
        spectra, wavelengths, (1.559210, 1.868710, 2.450170), (7, 7, 7)
    )

    lcpindex2 = rb1690 * 0.2 + rb1750 * 0.2 + rb1810 * 0.3 + rb1870 * 0.3
    lcpindex2[lcpindex2 < 0] = 0
    return lcpindex2


def hcpindex2_calculation(
    spectra: np.ndarray,
    wavelengths: tuple[float, ...],
) -> np.ndarray:
    """Calculate the HCPINDEX2 summary parameter across an image.
    HCPINDEX2 used to identify presence of high-calcium pyroxenes.
    Negative values are clipped to 0.

    Parameters
    ----------
    spectra : np.ndarray
        Spectra to calculate HCPINDEX2 for.
        Shape (n_spectra, n_wavelengths)
    wavelengths : tuple[float, ...]
        Wavelengths corresponding to spectra.
        Shape (n_wavelengths,)

    Returns
    -------
    hcpindex2 : np.ndarray
        HCPINDEX2 values for each spectra.
        Shape (n_spectra,)

    References
    ----------
    1. Viviano-Beck CE, Seelos FP, Murchie SL, Kahn EG, Seelos KD, Taylor HW,
    et al. Revised CRISM spectral parameters and summary products based on the
    currently detected mineral diversity on Mars.
    Journal of Geophysical Research: Planets. 2014;119(6):1403-31.
    """
    rb2120 = _relative_band_depth_calculation(
        spectra, wavelengths, (1.809390, 2.119480, 2.529510), (7, 5, 7)
    )
    rb2140 = _relative_band_depth_calculation(
        spectra, wavelengths, (1.809390, 2.139300, 2.529510), (7, 7, 7)
    )
    rb2230 = _relative_band_depth_calculation(
        spectra, wavelengths, (1.809390, 2.231820, 2.529510), (7, 7, 7)
    )
    rb2250 = _relative_band_depth_calculation(
        spectra, wavelengths, (1.809390, 2.251650, 2.529510), (7, 7, 7)
    )
    rb2430 = _relative_band_depth_calculation(
        spectra, wavelengths, (1.809390, 2.430300, 2.529510), (7, 7, 7)
    )
    rb2460 = _relative_band_depth_calculation(
        spectra, wavelengths, (1.809390, 2.456790, 2.529510), (7, 7, 7)
    )

    hcpindex2 = (
        rb2120 * 0.1
        + rb2140 * 0.1
        + rb2230 * 0.15
        + rb2250 * 0.3
        + rb2430 * 0.2
        + rb2460 * 0.15
    )
    hcpindex2[hcpindex2 < 0] = 0
    return hcpindex2


def cindex2_calculation(
    spectra: np.ndarray,
    wavelengths: tuple[float, ...],
) -> np.ndarray:
    """Calculate the CINDEX2 summary parameter across an image.
    CINDEX2 used to identify presence of carbonates.
    Negative values are clipped to 0.

    Parameters
    ----------
    spectra : np.ndarray
        Spectra to calculate HCPINDEX2 for.
        Shape (n_spectra, n_wavelengths)
    wavelengths : tuple[float, ...]
        Wavelengths corresponding to spectra.
        Shape (n_wavelengths,)

    Returns
    -------
    cindex2 : np.ndarray
        CINDEX2 values for each spectra.
        Shape (n_spectra,)

    References
    ----------
    1. Viviano-Beck CE, Seelos FP, Murchie SL, Kahn EG, Seelos KD, Taylor HW,
    et al. Revised CRISM spectral parameters and summary products based on the
    currently detected mineral diversity on Mars.
    Journal of Geophysical Research: Planets. 2014;119(6):1403-31.
    """
    a, b = _wavelength_weights(3.450310, 3.610040, 3.876730)
    # 1 / band depth as applying inverse lever rule 
    # (looking for convexity rather than concavity). See
    cindex2 = 1 - 1 / (
        _band_depth_calculation(
            spectra, wavelengths, (3.450310, 3.610040, 3.876730), (9, 11, 7)
        )
    )
    return cindex2


IMPLEMENTED_SUMMARY_PARAMETERS = {
    "hyd_femg_clay_index": hyd_femg_clay_index_calculation,
    "d2300": d2300_calculation,
    "bd1750": bd1750_calculation,
    "bd175": bd175_calculation,
    "OLINDEX3": olindex3_calculation,
    "LCPINDEX2": lcpindex2_calculation,
    "HCPINDEX2": hcpindex2_calculation,
    "CINDEX2": cindex2_calculation,
}
