import numpy as np

def wavelength_weights(short_wavelength: float, center_wavelength: float, long_wavelength: float) -> tuple[float, float]:
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
    b = (center_wavelength - short_wavelength)/(long_wavelength - short_wavelength)
    a = 1 - b
    return a, b

def interpolated_center_wavelength_reflectance(short_ref, bd_wavelengths, long_ref,):
    """
    Internal function to calculate the centre wavelength reflectance of spectra.

    Parameters
    ----------
    short_ref : np.ndarray
        Median reflectance of kernel centred at short wavelength.
        Shape (n_spectra,)
    bd_wavelengths: tuple
        Wavelengths to calculate the centre wavelength reflectance for.
        (short_wavelength, center_wavelength, long_wavelength)
    long_ref : np.ndarray
        Median reflectance of kernel centred at long wavelength.
        Shape (n_spectra,)
    """
    a, b = wavelength_weights(bd_wavelengths[0], bd_wavelengths[1], bd_wavelengths[2])
    return (a*short_ref + b*long_ref)

def band_depth_calculation(spectra, all_wavelengths, bd_wavelengths, kernel_sizes):
    """
    Internal method to calculate the band depth for a given set of wavelengths.

    Parameters
    ----------
    spectra : np.ndarray
        Spectra to calculate band depth for. 
        Shape (n_spectra, n_wavelengths)
    all_wavelengths : np.ndarray
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

    kernel_sizes = [kernel_size//2 for kernel_size in kernel_sizes]

    short_ref = np.median(spectra[:, all_wavelengths.index(short_wavelength)-kernel_sizes[0]:all_wavelengths.index(short_wavelength)+kernel_sizes[0]+1], axis=1)
    center_ref = np.median(spectra[:, all_wavelengths.index(center_wavelength)-kernel_sizes[1]:all_wavelengths.index(center_wavelength)+kernel_sizes[1]+1], axis=1)
    long_ref = np.median(spectra[:, all_wavelengths.index(long_wavelength)-kernel_sizes[2]:all_wavelengths.index(long_wavelength)+kernel_sizes[2]+1], axis=1)

    interpolated_center_ref = interpolated_center_wavelength_reflectance(short_ref, bd_wavelengths, long_ref)
    band_depth = center_ref / interpolated_center_ref
    return band_depth

def hyd_femg_clay_index_calculation(spectra, wavelengths):
    """
    Calculate the summary parameter Hydrated Fe/Mg Clay Index across an image. 
    Index from [1].

    Parameters
    ----------
    spectra : np.ndarray
        Spectra to calculate Hydrated Fe/Mg Clay Index for. 
        Shape (n_spectra, n_wavelengths)
    wavelengths : np.ndarray
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

    bd14 = 1 - band_depth_calculation(spectra, wavelengths, (1.33578, 1.41459, 1.55264), (5, 3, 5))
    bd14[bd14 < 0] = 0
    bd19 = 1 - band_depth_calculation(spectra, wavelengths, (1.86212, 1.92146, 2.07985), (5, 5, 5))
    bd19[bd19 < 0] = 0
    bd229 = 1 - band_depth_calculation(spectra, wavelengths, (2.15251, 2.28472, 2.34426), (3, 5, 3))
    bd229[bd229 < 0] = 0
    bd231 = 1 - band_depth_calculation(spectra, wavelengths, (2.15251, 2.30456, 2.34426), (3, 3, 3))
    bd231[bd231 < 0] = 0
    bd232 = 1 - band_depth_calculation(spectra, wavelengths, (2.15251, 2.32441, 2.34426), (3, 3, 3))
    bd232[bd232 < 0] = 0
    bd238 = 1 - band_depth_calculation(spectra, wavelengths, (2.34426, 2.38396, 2.4303), (3, 3, 3))
    bd238[bd238 < 0] = 0
    bd240 = 1 - band_depth_calculation(spectra, wavelengths, (2.34426, 2.3972, 2.4303), (3, 3, 3))
    bd240[bd240 < 0] = 0

    femg_clays = bd14 + bd19 + bd229 + bd231 + bd232 + bd238 + bd240

    return femg_clays


def d2300_calculation(spectra, wavelengths):
    """
    Calculate the dropoff at 2300nm across an image.
    Highlights Mg,Fe-OH minerals, as well as Mg-Carbonates, and CO2 ice.

    Parameters
    ----------
    spectra : np.ndarray
        Spectra to calculate dropoff at 2300nm for. 
        Shape (n_spectra, n_wavelengths)
    wavelengths : np.ndarray
        Wavelengths corresponding to spectra.
        Shape (n_wavelengths,)

    Returns
    -------
    d2300 : np.ndarray
        Dropoff at 2300nm values for each spectra.
        Shape (n_spectra,)
    """
    d2300 = np.zeros(spectra.shape[0])


    bd2290 = band_depth_calculation(spectra, wavelengths, (1.81598, 2.30456, 2.52951), (5, 3, 5))
    bd2320 = band_depth_calculation(spectra, wavelengths, (1.81598, 2.32441, 2.52951), (5, 3, 5))
    bd2330 = band_depth_calculation(spectra, wavelengths, (1.81598, 2.23182, 2.52951), (5, 3, 5))
    bd2120 = band_depth_calculation(spectra, wavelengths, (1.81598, 2.11948, 2.52951), (5, 5, 5))
    bd2170 = band_depth_calculation(spectra, wavelengths, (1.81598, 2.17233, 2.52951), (5, 5, 5))
    bd2210 = band_depth_calculation(spectra, wavelengths, (1.81598, 2.21199, 2.52951), (5, 5, 5))

    d2300 = 1 - ((bd2290 + bd2320 + bd2330)/(bd2120 + bd2170 + bd2210))    

    return d2300

def bd1750_calculation(spectra, wavelengths):
    """
    Calculate the summary parameter BD1750 across an image. 
    BD1750 used to identify presence of absorption feature at 1.75um, present in Alunite and Gypsum.

    Parameters
    ----------
    spectra : np.ndarray
        Spectra to calculate BD1750 for. 
        Shape (n_spectra, n_wavelengths)
    wavelengths : np.ndarray
        Wavelengths corresponding to spectra.
        Shape (n_wavelengths,)
    
    Returns
    -------
    bd1750 : np.ndarray
        BD1750 values for each spectra.
        Shape (n_spectra,)
    """
    bd1750 = np.zeros(spectra.shape[0])

    bd1750 = 1 - band_depth_calculation(spectra, wavelengths, (1.55264, 1.75009, 1.81598), (1,1,1))
    bd1750[bd1750 < 0] = 0

    return bd1750
    
def bultel_bd175(spectra, wavelengths):
    
    bd175 = np.ones(spectra.shape[0])
    lambda_c_1_idx = wavelengths.index(1.75009)
    lambda_c_2_idx = wavelengths.index(1.75668)
    lambda_l_idx = wavelengths.index(1.77644)
    lambda_s_idx = wavelengths.index(1.69082)

    bd175 = bd175 - ((spectra[:, lambda_c_1_idx] + spectra[:, lambda_c_2_idx])/(spectra[:, lambda_s_idx] + spectra[:, lambda_l_idx]))
    return bd175