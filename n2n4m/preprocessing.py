import os
import pandas as pd
import warnings
from sklearn.metrics import r2_score
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from n2n4m.wavelengths import ALL_WAVELENGTHS, PLEBANI_WAVELENGTHS
import n2n4m.utils as utils

LABEL_COLS = ["Image_Name", "Pixel_Class", "Coordinates"]


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the JSON dataset from the given path.

    Parameters
    ----------
    path : str
        Path to the dataset. Must be a JSON file.

    Returns
    -------
    dataset : pd.DataFrame
        The dataset.
    """
    if not os.path.exists(path):
        raise FileNotFoundError("The given path does not exist.")
    if not path.endswith(".json"):
        raise ValueError("The given path must be a JSON file.")
    return pd.read_json(path, dtype={"Image_Name": "string"})


def expand_dataset(
    dataset: pd.DataFrame, bands: tuple = ALL_WAVELENGTHS
) -> pd.DataFrame:
    """
    Convert the spectrum column into a column per wavelength value.

    Parameters
    ----------
    dataset : pd.DataFrame
        The dataset to expand.
        Requires each spectra to be in a list under the column "Spectrum".
    bands : tuple
        The centrepoint wavelengths of the bands to expand the dataset to.
        Must match length of spectra.
        Default: wavelengths.ALL_WAVELENGTHS

    Returns
    -------
    expanded_dataset : pd.DataFrame
        The expanded dataset.
    """

    def expanded_array_column(row):
        return pd.Series(row["Spectrum"])

    expanded_dataset = dataset.apply(expanded_array_column, axis=1)
    expanded_dataset.columns = [str(wavelength) for wavelength in bands]
    dataset = pd.concat([dataset, expanded_dataset], axis=1)
    dataset = dataset.drop(columns=["Spectrum"])
    return dataset


def drop_bad_bands(dataset: pd.DataFrame, bands_to_keep: tuple = PLEBANI_WAVELENGTHS):
    """
    Drop any bands with consistently bad pixels.
    Default drops 1.00135, 1.0079, > 3.9167, and between 2.66816 and 2.80697 inclusive
    Parameters
    ----------
    dataset : pd.DataFrame
        The dataset to drop the bad wavelengths from.
    bands_to_keep : tuple
        The wavelengths to keep.

    Returns
    -------
    dataset : pd.DataFrame
        The dataset with the bad wavelengths dropped.
    """
    current_cols = dataset.columns
    current_cols = [col for col in current_cols if col not in LABEL_COLS]
    missing_wavelengths = [
        str(band) for band in current_cols if float(band) not in bands_to_keep
    ]
    dataset = dataset.drop(columns=missing_wavelengths, axis=1)

    return dataset


def detect_bad_values(dataset: pd.DataFrame, threshold: float = 1.0) -> bool:
    """
    Detect whether any bad values are present in the numerical data of passed dataset.

    Parameters
    ----------
    dataset : pd.DataFrame
        The dataset to detect bad values in.
    threshold : float
        The threshold to use to detect bad values.
        Default: 1.

    Returns
    -------
    bool
        Whether bad values are present in the dataset.
    """
    data = dataset.drop(columns=LABEL_COLS)
    label_data = dataset[LABEL_COLS]
    data[data > threshold] = np.nan
    if data.isna().any().any():
        return True
    else:
        return False


def impute_column_mean(dataset: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    """
    Impute any bad values in the dataset with the mean of the column.
    Dataset modified in place.

    Parameters
    ----------
    dataset : pd.DataFrame
        The dataset to impute the bad values in.
    threshold : float
        The threshold to use to detect bad values.
        Default: 1.

    Returns
    -------
    dataset : pd.DataFrame
        The dataset with the bad values imputed.
    """
    data = dataset.drop(columns=LABEL_COLS)
    label_data = dataset[LABEL_COLS]
    data[data > threshold] = np.nan
    data = data.fillna(data.mean())
    dataset = pd.concat([label_data, data], axis=1)
    return dataset


def impute_bad_values(dataset: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    """
    Impute any bad values in the dataset.
    Dataset modified in place.

    Uses a 3 level strategy:
    1. If good values are present in the same class in the same image, impute using the mean of that band for those pixels.
    2. If good values are present in the same image, impute using the mean of that band for those pixels.
    3. If good values are present in the dataset, impute using the mean of that band for those pixels.

    Parameters
    ----------
    dataset : pd.DataFrame
        The dataset to impute the bad values in.
    threshold : float
        The threshold to use to detect bad values.
        Default: 1

    Returns
    -------
    dataset : pd.DataFrame
        The dataset with the bad values imputed.
    """
    dataset = utils.label_list_to_string(dataset)
    for image_name in dataset["Image_Name"].unique():
        for pixel_class in dataset[dataset["Image_Name"] == image_name][
            "Pixel_Class"
        ].unique():
            class_subset = dataset[
                (dataset["Image_Name"] == image_name)
                & (dataset["Pixel_Class"] == pixel_class)
            ]

            class_subset = impute_column_mean(class_subset, threshold=threshold)
            dataset.update(class_subset)

        if detect_bad_values(dataset[dataset["Image_Name"] == image_name]):
            dataset.update(
                impute_column_mean(
                    dataset[dataset["Image_Name"] == image_name], threshold=threshold
                )
            )
    if detect_bad_values(dataset):
        dataset.update(impute_column_mean(dataset, threshold=threshold))

    dataset = utils.label_string_to_list(dataset)
    return dataset


def get_linear_interp_spectra(
    spectra: np.ndarray,
    lower_bound: float = 1.91487,
    upper_bound: float = 2.08645,
    wavelengths: tuple = PLEBANI_WAVELENGTHS,
)-> np.ndarray:
    """
    Get the linear interpolation of the spectra between the lower and upper bounds.

    Parameters
    ----------
    spectra : np.ndarray
        The spectra to interpolate.
    lower_bound : float
        The lower bound wavelength.
        Default: 1.91487
    upper_bound : float
        The upper bound wavelength.
        Default: 2.08645
    wavelengths : tuple
        The wavelengths of the spectra.
        Default: PLEBANI_WAVELENGTHS

    Returns
    -------
    linear_interp : np.ndarray
        The linear interpolation of the spectra between the lower and upper bounds.
    """
    lower_bound_idx = wavelengths.index(lower_bound)
    upper_bound_idx = wavelengths.index(upper_bound)

    linear_interp = np.poly1d(
        np.polyfit(
            wavelengths[lower_bound_idx:upper_bound_idx],
            spectra[lower_bound_idx:upper_bound_idx],
            1,
        )
    )(wavelengths[lower_bound_idx:upper_bound_idx])
    return linear_interp


def detect_artefact(
    spectra: np.ndarray,
    lower_bound: float = 1.91487,
    upper_bound: float = 2.08645,
    wavelengths: tuple = PLEBANI_WAVELENGTHS,
    threshold: float = 0.6,
):
    """
    Detect whether an artefact is present in the spectra between the lower and upper bounds.

    Parameters
    ----------
    spectra : np.array
        The spectra to interpolate.
    lower_bound : float
        The lower bound wavelength.
        Default: 1.91487
    upper_bound : float
        The upper bound wavelength.
        Default: 2.08645
    wavelengths : tuple
        The wavelengths of the spectra.
        Default: PLEBANI_WAVELENGTHS

    Returns
    -------
    bool
        Whether an artefact is present in the spectra between the lower and upper bounds.
    """
    linear_interp = get_linear_interp_spectra(
        spectra, lower_bound, upper_bound, wavelengths
    )
    r2 = r2_score(
        linear_interp,
        spectra[wavelengths.index(lower_bound) : wavelengths.index(upper_bound)],
    )
    if r2 < threshold:
        return True
    else:
        return False


def impute_artefacts(
    spectra, lower_bound=1.91487, upper_bound=2.08645, wavelengths=PLEBANI_WAVELENGTHS
):
    """
    Impute the artefacts in the spectra between the lower and upper bounds.
    Uses the mean of spectra with no artefacts in the same mineral class to impute the artefacts.
    Fits the residual of the mean when removed from the continuum to the artefact spectra continuum.

    Parameters
    ----------
    spectra : pd.DataFrame
        The spectra to interpolate.
    lower_bound : float
        The lower bound wavelength.
        Default: 1.91487
    upper_bound : float
        The upper bound wavelength.
        Default: 2.08645
    wavelengths : tuple
        The wavelengths of the spectra.
        Default: PLEBANI_WAVELENGTHS

    Returns
    -------
    spectra_copy : pd.DataFrame
        The interpolated spectra.
    """
    spectra["Pixel_Class"] = spectra["Pixel_Class"].apply(lambda x: x[0])
    spectra_copy = spectra.copy(deep=True)
    lower_bound_idx = wavelengths.index(lower_bound)
    upper_bound_idx = wavelengths.index(upper_bound)
    band_column_headers = [str(wavelength) for wavelength in wavelengths]
    for mineral in spectra["Pixel_Class"].unique():
        print(f"Mineral class: {mineral}")
        mineral_spectra = spectra[spectra["Pixel_Class"] == mineral]
        mineral_spectra = mineral_spectra.drop(
            columns=["Pixel_Class", "Image_Name", "Coordinates"]
        ).values
        artefact_mask = [detect_artefact(spectra) for spectra in mineral_spectra]
        if sum(artefact_mask) == 0:
            warnings.warn(f"No artefacts detected in {mineral} spectra, skipping.")
            continue
        if sum(artefact_mask) == len(artefact_mask):
            warnings.warn(f"All {mineral} spectra contain artefacts, skipping.")
            continue
        print(
            f"Number of spectra: {len(artefact_mask)}\nNumber of artefacts: {sum(artefact_mask)}\nPercentage of artefacts: {sum(artefact_mask)/len(artefact_mask)}"
        )
        no_artefact_mineral_spectra = mineral_spectra[~np.array(artefact_mask)]
        average_no_artefact_mineral = np.mean(no_artefact_mineral_spectra, axis=0)
        no_artefact_linear_interp = np.poly1d(
            np.polyfit(
                wavelengths[lower_bound_idx:upper_bound_idx],
                average_no_artefact_mineral[lower_bound_idx:upper_bound_idx],
                1,
            )
        )(wavelengths[lower_bound_idx:upper_bound_idx])
        no_artefact_residual = (
            average_no_artefact_mineral[lower_bound_idx:upper_bound_idx]
            - no_artefact_linear_interp
        )
        for idx, spectrum_mask in enumerate(zip(mineral_spectra, artefact_mask)):
            spectrum = spectrum_mask[0]
            mask = spectrum_mask[1]
            if mask:
                artefact_linear_interp = np.poly1d(
                    np.polyfit(
                        wavelengths[lower_bound_idx:upper_bound_idx],
                        spectrum[lower_bound_idx:upper_bound_idx],
                        1,
                    )
                )(wavelengths[lower_bound_idx:upper_bound_idx])
                mineral_spectra[idx, lower_bound_idx:upper_bound_idx] = (
                    artefact_linear_interp + no_artefact_residual
                )

        spectra_copy.loc[
            spectra_copy["Pixel_Class"] == mineral, band_column_headers
        ] = mineral_spectra
    spectra_copy["Pixel_Class"] = spectra_copy["Pixel_Class"].apply(lambda x: [x])
    return spectra_copy


def generate_noisy_pixels(dataset, random_seed=False):
    """
    Generate the noisy pixels for the dataset.

    Parameters
    ----------
    dataset : pd.DataFrame
        The dataset to generate the noisy pixels for.
    random_seed : int
        The random seed to use for the noise generation. If False, no random seed is used.
        Default False.

    Returns
    -------
    dataset : pd.DataFrame
        New dataframe with the noisy pixels.
        Identical shape to original dataset, but with "_noisy" appended to the column name.
    """
    if random_seed:
        np.random.seed(random_seed)

    dataset = dataset.apply(
        lambda x: x
        + np.random.normal(
            0, np.random.uniform(0, np.random.uniform(0, 0.02, len(x)), len(x)), len(x)
        ),
        axis=0,
    )

    dataset_columns = dataset.columns
    dataset_columns = [x + "_noisy" for x in dataset_columns]
    dataset.columns = dataset_columns
    return dataset


def train_test_split(dataset, bland_pixels=False):
    """
    Split the dataset into a training set and testing set.
    Sets are not random, but designed to ensure a good split of classes,
    geographic locations, and a sufficient quantity of pixels.

    Parameters
    ----------
    dataset : pd.DataFrame
        The dataset to split.
    bland_pixels : bool
        Whether images containing only bland pixels are included in the dataset.
        Default False.

    Returns
    -------
    train_set : pd.DataFrame
        The training set.
    test_set : pd.DataFrame
        The testing set.
    """
    # fmt: off
    test_set_image_names = [
        "0A053", "0634B", "09036", "050F2", "0A546", "081CF", "02FC5", "0454E",
        "08F68", "0285A", "0863E", "0A425", "19538", "19DAA", "064D9"
       ]
    if bland_pixels:
        test_set_image_names = test_set_image_names + [
            "0289E", "028BA", "03266", "040FF", "04185", "042AA", "047A3",
            "048B2", "05483", "05850", "058A3", "05C5E", "066A4", "07BC8",
            "07C95", "082EE", "0860C", "088D0", "092F7", "09365", "095EE",
            "0BA8D", "0BAB3", "0BADC", "0BB36", "0BBFF", "0BDBB", "0BDBE",
            "0BDD9", "0BEC0", "0BECE", "0BEDF", "0BEF2", "0BFA6", "0BFC2",
            "0BFF1", "0C008", "0C08D", "0C0AF", "0C0EF", "0C141", "0C23D",
            "0C256", "0C280", "0C441", "0C4BB", "0C518", "0C613", "0C620",
            "0C712", "0C75D", "0C95A", "0C968", "0C9F7", "0C9FB", "0CAB2",
        ]
    # fmt: on
    train_set = dataset[~dataset["Image_Name"].isin(test_set_image_names)]
    test_set = dataset[dataset["Image_Name"].isin(test_set_image_names)]
    return train_set, test_set


def train_validation_split(dataset, bland_pixels=False):
    """
    Split the train set into a train set and validation set.
    Sets are not random, but designed to ensure a good split of classes,
    geographic locations, and a sufficient quantity of pixels.

    Parameters
    ----------
    dataset : pd.DataFrame
        The dataset to split.
    bland_pixels : bool
        Whether images containing only bland pixels are included in the dataset.
        Default False.

    Returns
    -------
    train_set : pd.DataFrame
        The training set.
    validation_set : pd.DataFrame
        The validation set.
    """
    # fmt: off
    validation_set_image_names = [
        "093BE", "0B252", "02885", "043EC", "20BF9", "21D02", "0AA7D", 
        "20AE1", "0750A", "09971",
    ]
    if bland_pixels:
        validation_set_image_names = validation_set_image_names + [
            "095FE", "09ABE", "09ACE", "09B66", "09D44", "09D96", "09E4C",
            "09E5D", "0A0D1", "0A253", "0A377", "0A5AA", "0ABCB", "0AC95",
            "0ACE6", "0B2A2", "0B6A2", "0B6C5", "0B6F1", "0B977",
        ]
    # fmt: on

    train_set = dataset[~dataset["Image_Name"].isin(validation_set_image_names)]
    validation_set = dataset[dataset["Image_Name"].isin(validation_set_image_names)]
    return train_set, validation_set


def split_features_targets_anciliary(dataset):
    """
    Function to split a dataset into features, targets, and anciliary information (image name, pixel coordinates, class label)

    Parameters
    ----------
    dataset : pd.DataFrame
        The dataset to split.

    Returns
    -------
    features : pd.DataFrame
        The features of the dataset.
    targets : pd.DataFrame
        The targets of the dataset.
    anciliary_data : pd.DataFrame
        The anciliary information of the dataset.
    """
    ancillary_data = dataset[["Image_Name", "Coordinates", "Pixel_Class"]]
    dataset = dataset.drop(columns=["Image_Name", "Coordinates", "Pixel_Class"])
    features = dataset.iloc[:, int(len(dataset.columns) / 2) :]
    targets = dataset.iloc[:, : int(len(dataset.columns) / 2)]
    return features, targets, ancillary_data


def standardise(dataset, method="StandardScaler", scaler=None):
    """
    Standardise a dataset. If a scaler is given, the dataset is transformed using the given scaler, else new scaler is fitted and returned.
    Supported methods include:
    - StandardScaler
    - MinMaxScaler
    - RobustScaler

    Parameters
    ----------
    train_dataset : pd.DataFrame
    method : str
        The method to use for standardisation. Default is StandardScaler.
    scaler : sklearn.preprocessing object
        The scaler to use. If None, a new scaler is fitted and returned.

    Returns
    -------
    scaled_dataset : pd.DataFrame
        The scaled dataset.
    scaler : sklearn.preprocessing object
        The fitted scaler object.
    """

    if scaler == None:
        if method == "StandardScaler":
            scaler = StandardScaler()
        elif method == "MinMaxScaler":
            scaler = MinMaxScaler()
        elif method == "RobustScaler":
            scaler = RobustScaler()
        else:
            raise ValueError("The given method is not supported.")
        scaler.fit(dataset)

    scaled_dataset = pd.DataFrame(
        scaler.transform(dataset), columns=dataset.columns, index=dataset.index
    )
    return scaled_dataset, scaler


def inverse_standardise(dataset, scaler):
    """
    Inverse standardise a dataset using the given scaler.

    Parameters
    ----------
    dataset : pd.DataFrame or np.array
        The dataset to inverse standardise.
    scaler : sklearn.preprocessing object
        The fitted scaler object.

    Returns
    -------
    dataset : pd.DataFrame or np.array
        The inverse standardised dataset.
    """

    if type(dataset) == pd.DataFrame:
        dataset_columns = dataset.columns
        if len(dataset) == 1:
            dataset = scaler.inverse_transform(dataset.to_numpy().reshape(1, -1))
        else:
            dataset = scaler.inverse_transform(dataset.to_numpy())
        dataset = pd.DataFrame(dataset, columns=dataset_columns)

    elif type(dataset) == pd.Series:
        dataset_index = dataset.index
        dataset = scaler.inverse_transform(dataset.to_numpy().reshape(1, -1))
        dataset = pd.Series(dataset[0], index=dataset_index)

    elif type(dataset) == np.ndarray:
        if dataset.ndim == 1:
            dataset = scaler.inverse_transform(dataset.reshape(1, -1))
        else:
            dataset = scaler.inverse_transform(dataset)

    elif type(dataset) == torch.Tensor:
        if dataset.dim() == 1:
            dataset = scaler.inverse_transform(dataset.reshape(1, -1))
        else:
            dataset = scaler.inverse_transform(dataset)
        dataset = torch.from_numpy(dataset)
    else:
        raise TypeError("The given dataset type is not supported.")

    return dataset
