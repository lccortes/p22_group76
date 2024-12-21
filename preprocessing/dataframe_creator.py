import pandas as pd
import os
import re

# Regular expression pattern to match the file format and extract parts
pattern = re.compile(
    r"(?P<name>[^_]+)_(?P<cell_type>[^_]+)_well(?P<well_number>\d+)_(?P<location_number>[^_]+)_(?P<tag>s\d+z\d+c\d+)_(?P<data_type>ORG(?:_mask)?)\.(?P<file_format>tif{1,2})"
)

def process_files_in_directory(data: list[dict[str, str]], directory: str, has_subfolders: bool = True) -> list[dict[str, str]]:
    if has_subfolders:
        for subfolder in os.listdir(directory):
            subfolder_path = os.path.join(directory, subfolder)
            if os.path.isdir(subfolder_path):
                for file in os.listdir(subfolder_path):
                    file_path = os.path.join(subfolder_path, file)
                    match_result = pattern.match(file)
                    if match_result:
                        # Extract data from filepath and add to the list
                        row = match_result.groupdict()
                        row["path"] = file_path
                        row["filename"] = file
                        row["subfolder"] = subfolder
                        data.append(row)
    else:
        # Process files directly within the directory (e.g. for `data/masks`)
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            match_result = pattern.match(file)
            if match_result:
                row = match_result.groupdict()
                row["path"] = file_path
                row["filename"] = file
                row["subfolder"] = None  # No subfolder for data/masks
                data.append(row)
    return data


def add_tag_split(df: pd.DataFrame) -> pd.DataFrame:
    # Split the `tag` column into separate columns
    df["sample_site"] = df["tag"].str.extract(r"s(\d+)").astype(int)
    df["z_position"] = df["tag"].str.extract(r"z(\d+)").astype(int)
    df["channel"] = df["tag"].str.extract(r"c(\d+)").astype(int)
    return df


def add_mask_indicator_column(df: pd.DataFrame) -> pd.DataFrame:
    df["is_mask"] = df["path"].str.contains("mask")
    return df

def add_mask_path_column(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a column to the dataframe with corresponding mask filepaths for each brightfield image row"""

    # create a table with just masks where is_mask is True
    df_only_masks = df[df["is_mask"] == True]
    # create a table with just brightfields where is_mask is False
    df_only_brightfield = df[df["is_mask"] == False]

    # join df_masks to df_no_masks by sample_id and well_id
    # the new df will have all columns from df_no_masks and just the colum path from df_masks that will be renamed to mask_path
    df_mask_paths = df_only_brightfield.merge(df_only_masks[["sample_site", "well_number", "path"]], on=["sample_site", "well_number"], how="left")

    df_mask_paths.rename(columns={"path_x": "path"}, inplace=True)
    df_mask_paths.rename(columns={"path_y": "mask_path"}, inplace=True)
    
    #append masks to add_path as final table, the column mask_path should be filled with nulls
    df_final = pd.concat([df_mask_paths, df_only_masks], ignore_index=True)

    return df_final
    

def do_preprocessing() -> pd.DataFrame:
    data = []
    # Process brightfield and masks directories
    data = process_files_in_directory(data, "./data/brightfield", has_subfolders=True)
    data = process_files_in_directory(data, "./data/masks", has_subfolders=False)

    # Create DataFrame from the list of dictionaries
    df = pd.DataFrame(data)
    df = add_tag_split(df)
    df = add_mask_indicator_column(df)
    df = add_mask_path_column(df)
    return df


def save_dataframe(df: pd.DataFrame, path: str):
    """Saves preprocessed dataframe to csv

    :param df: pandas Dataframe to save
    :type df: pd.DataFrame
    :param path: file path to define where to save the resulting CSV file
    :type path: str
    """

    df.to_csv(path, sep=";", encoding="utf-8")


def load_dataframe_from_csv(path: str) -> pd.DataFrame:
    """Loads preprocessed csv to dataframe

    :param path: Path to the csv file to load dataframe from
    :type path: str
    :returns: dataframe from csv
    :rtype: pd.DataFrame
    """

    df = pd.read_csv(path, sep=";", encoding="utf-8")
    return df
