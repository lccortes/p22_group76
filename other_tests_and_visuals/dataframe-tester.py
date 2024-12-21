import marimo

__generated_with = "0.9.17"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __():
    import pandas as pd
    import os
    import re
    return os, pd, re


@app.cell
def __(os, pd, re):
    # Regular expression pattern to match the file format and extract parts
    pattern = re.compile(
        r"(?P<name>[^_]+)_(?P<cell_type>[^_]+)_well(?P<well_number>\d+)_(?P<location_number>[^_]+)_(?P<tag>s\d+z\d+c\d+)_(?P<data_type>ORG(?:_mask)?)\.(?P<file_format>tif{1,2})"
    )

    # List to store each row of extracted information
    data: list[dict[str, str]] = []


    def process_files_in_directory(directory: str, has_subfolders: bool = True):
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


    def append_tag_split(df: pd.DataFrame) -> pd.DataFrame:
        # Split the `tag` column into separate columns
        df["sample_site"] = df["tag"].str.extract(r"s(\d+)").astype(int)
        df["z_position"] = df["tag"].str.extract(r"z(\d+)").astype(int)
        df["channel"] = df["tag"].str.extract(r"c(\d+)").astype(int)
        return df


    def add_mask_indicator_column(df: pd.DataFrame) -> pd.DataFrame:
        df["is_mask"] = df["path"].str.contains("mask")
        return df


    def do_preprocessing() -> pd.DataFrame:
        # Process brightfield and masks directories
        process_files_in_directory("./data/brightfield", has_subfolders=True)
        process_files_in_directory("./data/masks", has_subfolders=False)

        # Create DataFrame from the list of dictionaries
        df = pd.DataFrame(data)
        df = append_tag_split(df)
        df = add_mask_indicator_column(df)
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
    return (
        add_mask_indicator_column,
        append_tag_split,
        data,
        do_preprocessing,
        load_dataframe_from_csv,
        pattern,
        process_files_in_directory,
        save_dataframe,
    )


@app.cell
def __(do_preprocessing):
    df = do_preprocessing()

    # Display the DataFrame
    df
    return (df,)


@app.cell
def __(df):
    masks = df[df["is_mask"] == True]
    brightfield = df[df["is_mask"] == False]

    masks = masks.rename(columns={"path": "corresponding_mask_path"})
    df2 = brightfield.merge(masks, on=["well_number", "sample_site"], how="left").copy()
    df2.head()
    return brightfield, df2, masks


@app.cell
def __(masks):
    mask_duplicates = masks[masks.duplicated(subset=["well_number", "sample_site"], keep="last")]
    mask_duplicates
    return (mask_duplicates,)


@app.cell
def __(df, mo):
    df["corresponding_mask"] = ""

    for i, row in mo.status.progress_bar(list(df[df['is_mask'] == False].iterrows())):
        # Find the row with the same 'well' and 'sample' and is_mask == False
        corresponding_row = df[(df['well_number'] == row['well_number']) & 
                               (df['sample_site'] == row['sample_site']) & 
                               (df['is_mask'] == True)]
        # print(corresponding_row)
        if not corresponding_row.empty:
            # Update corresponding row with True in the 'corresponding_mask' column
            df.at[i, 'corresponding_mask'] = df.at[corresponding_row.index[0], "path"]

    df
    return corresponding_row, i, row


@app.cell
def __(brightfield, df, df2, masks):
    len(df), len(df2), len(brightfield), len(masks)
    return


@app.cell
def __(df):
    first_index, first = next(df[df['is_mask'] == False].iterrows())
    corresponding_row_first = df[(df['well_number'] == first['well_number']) & 
                            (df['sample_site'] == first['sample_site']) & 
                            (df['is_mask'] == True)]
    print(corresponding_row_first)
    # first
    print(first_index)
    df.at[first_index, 'corresponding_mask'] =  df.at[corresponding_row_first.index[0], "path"]
    # df.at[first_index, 'corresponding_mask'] = str(corresponding_row_first["path"])
    df
    return corresponding_row_first, first, first_index


@app.cell
def __(corresponding_row_first, df):
    df.at[corresponding_row_first.index[0], "path"]
    return


@app.cell
def __(df):
    unique_corresponding_masks = df["corresponding_mask"].unique()
    unique_corresponding_masks
    return (unique_corresponding_masks,)


if __name__ == "__main__":
    app.run()
