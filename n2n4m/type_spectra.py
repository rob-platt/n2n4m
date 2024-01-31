import os
import pandas as pd

class_labels = {  # From Plebani et al. 2022 https://github.com/Banus/crism_ml/blob/master/crism_ml/lab.py
    1: "CO2 Ice",  # co2_ice
    2: "H2O Ice",  # h20_ice
    3: "Gypsum",  # gypsum
    4: "Ferric Hydroxysulfate",  # hydrox_fe_sulf
    5: "Hematite",  # hematite
    6: "Nontronite",  # fe_smectite
    7: "Saponite",  # mg_smectite
    8: "Prehnite",  # Prehnite Zeolite # prehnite
    9: "Jarosite",  # jarosite
    10: "Serpentine",  # serpentine
    11: "Alunite",  # alunite
    12: "Akaganeite",  # Fe Oxyhydroxysulfate # hydrox_fe_sulf
    13: "Ca/Fe CO3",  # Calcite, Ca/Fe carbonate  # fe_ca_carbonate
    14: "Beidellite",  # Al-smectite # al_smectite
    15: "Kaolinite",  # kaolinite
    16: "Bassanite",  # bassanite
    17: "Epidote",  # epidote
    18: "Montmorillonite",  # Al-smectite # al_smectite
    19: "Rosenite",  # Polyhydrated sulfate # poly_hyd_sulf
    20: "Mg Cl salt",  # Mg(ClO3)2.6H2O # Polyhydrated sulfate # poly_hyd_sulf
    21: "Halloysite",  # Kaolinite # kaolinite
    22: "Epsomite",  # Polyhydrated sulfate # poly_hyd_sulf
    23: "Illite/Muscovite",  # illite_muscovite
    24: "Margarite",  # Illite/Muscovite # illite_muscovite
    25: "Analcime",  # Zeolite # analcime
    26: "Monohydrated sulfate",  # Szomolnokite # mono_hyd_sulf
    27: "Opal 1",  # Opal # Hydrated silica # hydrated_silica
    28: "Opal 2",  # Opal-A # Hydrated silica # hydrated_silica
    29: "Iron Oxide Silicate Sulfate",  # Polyhydrated sulfate # poly_hyd_sulf
    30: "MgCO3",  # Magnesite # mg_carbonate
    31: "Chlorite",  # chlorite
    32: "Clinochlore",  # chlorite
    33: "Low Ca Pyroxene",  # lcp
    34: "Olivine Forsterite",  # mg_olivine
    35: "High Ca Pyroxene",  # hcp
    36: "Olivine Fayalite",  # fe_olivine
    37: "Chloride",  # chloride
    38: "Artifact",  # None
    39: "Neutral",  # None
}

type_spectra_labels = {  # From Plebani et al. 2022 https://github.com/Banus/crism_ml/blob/master/crism_ml/lab.py
    1: "co2_ice",
    2: "h2o_ice",
    3: "gypsum",
    4: "hydrox_fe_sulf",
    5: "hematite",
    6: "fe_smectite",
    7: "mg_smectite",
    8: "prehnite",
    9: "jarosite",
    10: "serpentine",
    11: "alunite",
    12: "hydrox_fe_sulf",
    13: "fe_ca_carbonate",
    14: "al_smectite",
    15: "kaolinite",
    16: "bassanite",
    17: "epidote",
    18: "al_smectite",
    19: "poly_hyd_sulf",
    20: "poly_hyd_sulf",
    21: "kaolinite",
    22: "poly_hyd_sulf",
    23: "illite_muscovite",
    24: "illite_muscovite",
    25: "analcime",
    26: "mono_hyd_sulf",
    27: "hydrated_silica",
    28: "hydrated_silica",
    29: "poly_hyd_sulf",
    30: "mg_carbonate",
    31: "chlorite",
    32: "chlorite",
    33: "lcp",
    34: "mg_olivine",
    35: "hcp",
    36: "fe_olivine",
    37: "chloride",
    38: "artifact",
    39: "neutral",
}

type_spectra_names = {  # From Plebani et al. 2022 https://github.com/Banus/crism_ml/blob/master/crism_ml/lab.py
    1: "CO2 Ice",
    2: "H2O Ice",
    3: "Gypsum",
    4: "Hydroxylated Fe Sulfate",
    5: "Hematite",
    6: "Fe Smectite",
    7: "Mg Smectite",
    8: "Prehnite",
    9: "Jarosite",
    10: "Sepentine",
    11: "Alunite",
    12: "Hydroxylated Fe Sulfate",
    13: "Fe/Ca Carbonate",
    14: "Al Smectite",
    15: "Kaolinite",
    16: "Bassanite",
    17: "Epidote",
    18: "Al Smectite",
    19: "Polyhydrated Sulfate",
    20: "Polyhydrated Sulfate",
    21: "Kaolinite",
    22: "Polyhydrated Sulfate",
    23: "Illite/Muscovite",
    24: "Illite/Muscovite",
    25: "Analcime",
    26: "Monohydrated Sulfate",
    27: "Hydrated Silica",
    28: "Hydrated Silica",
    29: "Polyhydrated Sulfate",
    30: "Mg Carbonate",
    31: "Chlorite",
    32: "Chlorite",
    33: "Low Ca Pyroxene",
    34: "Mg Olivine",
    35: "High Ca Pyroxene",
    36: "Fe Olivine",
    37: "Chloride",
    38: "Artifact",
    39: "Neutral/Bland",
}

CRISM_diagnostic_mineral_features = (
    {  # Collated from the CRISM MICA Files http://crism.jhuapl.edu/data/mica/
        "fe_olivine": [0.85, 1.12, 1.3],
        "mg_olivine": [0.84, 1.05, 1.25],
        "epidote": [1.54, 2.26, 2.35],
        "hcp": [0.99, 2.16],
        "lcp": [0.91, 1.95],
        "prehnite": [1.48, 1.90, 2.24, 2.29, 2.35, 2.47, 2.52],
        "al_smectite": [1.41, 1.91, 2.20],
        "chlorite": [1.41, 1.92, 2.0, 2.25, 2.35],
        "fe_smectite": [1.42, 1.91, 2.29, 2.39, 2.52],
        "illite_muscovite": [1.41, 1.93, 2.21, 2.35, 2.45],
        "kaolinite": [1.39, 1.41, 1.92, 2.16, 2.21, 2.38],
        "margarite": [1.41, 2.00, 2.20, 2.25, 2.35, 2.47],
        "mg_smectite": [1.41, 1.92, 2.31, 2.39],
        "serpentine": [1.39, 1.96, 2.11, 2.32, 2.51],
        "talc": [1.39, 1.91, 2.24, 2.31, 2.39, 2.47],
        "analcime": [1.42, 1.47, 1.78, 1.91, 2.52],
        "plagioclase": [1.32],
        "hematite": [0.55, 0.85],
        "chloride": [],
        "fe_ca_carbonate": [2.79, 3.35],
        "mg_carbonate": [1.91, 2.31, 2.51, 3.46, 3.84],
        "alunite": [1.34, 1.43, 1.48, 1.77, 2.16, 2.32, 2.45, 2.51],
        "bassanite": [1.43, 1.77, 1.92, 2.26, 2.49],
        "gypsum": [1.44, 1.49, 1.54, 1.75, 1.95, 2.21, 2.26, 2.48],
        "hydrox_fe_sulf": [1.48, 1.82, 1.99, 2.19, 2.23, 2.36],
        "jarosite": [0.90, 1.47, 1.85, 2.27],
        "mono_hyd_sulf": [1.63, 1.97, 2.14, 2.40],
        "poly_hyd_sulf": [1.43, 1.94, 2.43],
        "hydrated_silica": [1.42, 1.94, 2.21],
        "co2_ice": [1.44, 1.58, 1.99, 2.29, 2.35],
        "h2o_ice": [1.65],
    }
)

additional_mineral_features = (
    {  # Collated from the CRISM MICA Files http://crism.jhuapl.edu/data/mica/
        "fe_olivine": [],
        "mg_olivine": [],
        "epidote": [1.05],
        "hcp": [],
        "lcp": [],
        "prehnite": [],
        "al_smectite": [],
        "chlorite": [0.7, 0.89],
        "fe_smectite": [0.93, 2.23],
        "illite_muscovite": [],
        "kaolinite": [2.32],
        "margarite": [],
        "mg_smectite": [],
        "serpentine": [],
        "talc": [],
        "analcime": [0.97, 1.19, 2.13],
        "plagioclase": [],
        "hematite": [],
        "chloride": [],
        "fe_ca_carbonate": [2.79, 3.35],
        "mg_carbonate": [3.27],
        "alunite": [1.0, 1.27, 2.21, 2.43],
        "bassanite": [1.18],
        "gypsum": [1.00, 1.20],
        "hydrox_fe_sulf": [0.95, 2.16],
        "jarosite": [0.64, 1.54, 2.41, 2.46, 2.52],
        "mono_hyd_sulf": [],
        "poly_hyd_sulf": [0.98, 1.18],
        "hydrated_silica": [],
        "co2_ice": [],
        "h2o_ice": [1.65],
    }
)


def get_mineral_class(mineral_sample: pd.Series) -> str:
    return class_labels[mineral_sample["Pixel_Class"][0]]


def get_type_spectra_class(mineral_sample: pd.Series) -> str:
    return type_spectra_labels[mineral_sample["Pixel_Class"][0]]


def get_type_spectra_name(mineral_sample: pd.Series) -> str:
    return type_spectra_names[mineral_sample["Pixel_Class"][0]]


def read_type_spectra(filepath: str) -> pd.Series:
    """
    Read the type spectra from the file.

    Parameters
    ----------
    filepath : str
        The path to the type spectra file.

    Returns
    -------
    spectrum : pandas.Series
        The type spectra as a pandas series with index wavelengths and values reflectance.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File {filepath} not found.")
    data_file = open(filepath)
    wavelengths = []
    reflectance = []
    for line in data_file:
        values = line.split()
        wavelengths.append(values[0][:-1])
        reflectance.append(float(values[1][:-1].strip("'")))
    data_file.close()
    spectrum = pd.Series(reflectance, index=wavelengths)
    return spectrum


def get_type_spectra(
    mineral_sample: pd.Series,
    type_spectra_path: str = "../data/type_spectra",
) -> pd.Series:
    """
    From a mineral sample, load the relevant type spectra.

    Parameters
    ----------
    mineral_sample : pd.Series
        A mineral sample from the dataset.
    type_spectra_dir : str, optional
        Path to the directory containing the type spectra files.

    Returns
    -------
    wavelengths : list
        The wavelengths of the spectra.
    spectra : list
        The spectra.
    """
    if not os.path.isdir(type_spectra_path):
        raise IOError(
            f"Directory {type_spectra_path} not found.\nPlease download the type spectra from: https://pds-geosciences.wustl.edu/mro/mro-m-crism-4-typespec-v1/mrocr_8001/data/\n"
            f"Linux command: wget -e robots=off -r --cut-dirs=4 -np -nH https://pds-geosciences.wustl.edu/mro/mro-m-crism-4-typespec-v1/mrocr_8001/data/ -P ../data/type_spectra/"
        )
    if not isinstance(mineral_sample, pd.Series):
        raise TypeError(
            f"The mineral sample must be a pandas series not {type(mineral_sample)}"
        )
    type_spectra_filepath = f"{type_spectra_path}/crism_typespec_{get_type_spectra_class(mineral_sample)}.tab"
    return read_type_spectra(type_spectra_filepath)


def clip_type_spectra(mineral_sample: pd.Series, type_spectra: pd.Series) -> pd.Series:
    """
    Clip the type spectra to the same wavelengths as the mineral sample.

    Parameters
    ----------
    mineral_sample : pd.Series
        A mineral sample from the dataset.
    type_spectra : pd.Series
        The type spectra.

    Returns
    -------
    type_spectra : pd.Series
        The clipped type spectra.
    """
    return type_spectra[type_spectra.index.isin(mineral_sample.index)]
