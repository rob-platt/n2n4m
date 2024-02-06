import pandas as pd


def label_list_to_string(dataset: pd.DataFrame) -> pd.DataFrame:
    "Utility function to convert the label from being wrapped in a list to an int"
    try:
        dataset["Pixel_Class"] = dataset["Pixel_Class"].apply(lambda x: x[0])
    except TypeError:
        print("Pixel_Class labels are already in string format")
    return dataset


def label_string_to_list(dataset: pd.DataFrame) -> pd.DataFrame:
    "Utility function to convert the label from being an int to being wrapped in a list"
    try:
        dataset["Pixel_Class"] = dataset["Pixel_Class"].apply(lambda x: [x])
    except TypeError:
        print("Pixel_Class labels are already in list format")
    return dataset


def convert_coordinates_to_xy(dataset):
    "Utility function to convert from a Coordinates column/index to x and y columns/index"
    if type(dataset) != pd.DataFrame and type(dataset) != pd.Series:
        raise TypeError("Dataset must be a pandas DataFrame or Series")
    if type(dataset) == pd.Series:
        dataset["x"] = dataset["Coordinates"][0]
        dataset["y"] = dataset["Coordinates"][1]
        dataset = dataset.drop("Coordinates")
    else:
        dataset["x"] = dataset["Coordinates"].apply(lambda x: x[0])
        dataset["y"] = dataset["Coordinates"].apply(lambda x: x[1])
        dataset = dataset.drop(columns=["Coordinates"])
    return dataset


def convert_xy_to_coordinates(dataset):
    "Utility function to convert from x and y columns/index to a Coordinates column/index"
    if type(dataset) != pd.DataFrame and type(dataset) != pd.Series:
        raise TypeError("Dataset must be a pandas DataFrame or Series")
    if type(dataset) == pd.Series:
        dataset["Coordinates"] = [dataset["x"], dataset["y"]]
        dataset = dataset.drop(index=["x", "y"])
    else:
        dataset["Coordinates"] = dataset.apply(lambda x: [x["x"], x["y"]], axis=1)
        dataset = dataset.drop(columns=["x", "y"])
    return dataset


def find_breakpoint(wavelengths: tuple[float, ...], threshold: float = 0.01) -> int:
    """
    Identify any gaps in the wavelengths where there are bands missing.
    Useful for identifying where bands are dropped to avoid bad bands.
    Returns the index of the first band after the gap for easy slicing.
    If no gap, returns -0.
    ONLY FINDS THE FIRST GAP.

    Parameters
    ----------
    wavelengths : tuple[float, ...]
        List of wavelengths in the dataset.
    threshold : float, optional
        The maximum gap allowed between bands
        Default 0.01

    Returns
    -------
    breakpoint : int
        Index of the first bad after the gap.
        If no gap, returns -0.
    """
    for idx, band in enumerate(wavelengths):
        if band - wavelengths[idx - 1] > threshold:
            return idx
    return -0
