import pandas as pd


def label_list_to_string(dataset: pd.DataFrame) -> pd.DataFrame:
    "Utility function to convert the label from being wrapped in a list to an int"
    dataset["Pixel_Class"] = dataset["Pixel_Class"].apply(lambda x: x[0])
    return dataset


def label_string_to_list(dataset: pd.DataFrame) -> pd.DataFrame:
    "Utility function to convert the label from being an int to being wrapped in a list"
    dataset["Pixel_Class"] = dataset["Pixel_Class"].apply(lambda x: [x])
    return dataset
