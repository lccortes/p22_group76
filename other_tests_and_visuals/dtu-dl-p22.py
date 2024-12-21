import marimo

__generated_with = "0.9.17"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Gif from data""")
    return


@app.cell
def __():
    from PIL import Image, ImageDraw, ImageFont
    import os
    import glob
    return Image, ImageDraw, ImageFont, glob, os


@app.cell
def __(Image, ImageDraw, ImageFont, glob, mo, os):
    def create_gif_from_tifs(folder_path, output_gif_path, duration=500):
        # List all .tif files in the folder and sort them by filename
        tif_files = sorted(glob.glob(os.path.join(folder_path, "*.tif")))

        # Check if there are .tif files
        if not tif_files:
            raise ValueError("No .tif files found in the specified folder.")

        frames = []

        for file_path in mo.status.progress_bar(tif_files):
            # Open the image
            img = Image.open(file_path)
            # resize image to 2/3 to save file size
            img = img.resize(
                (int(img.size[0] * 1 / 3), int(img.size[1] * 1 / 3)),
                Image.Resampling.LANCZOS,
            )

            # Append filename below the image
            filename = os.path.basename(file_path)
            img_with_text = add_text_below_image(img, filename)

            frames.append(img_with_text)

        # Save as GIF
        print("Saving GIF...")
        frames[0].save(
            output_gif_path,
            quality=60,
            optimize=True,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
        )
        print(f"GIF created successfully at {output_gif_path}")


    def add_text_below_image(image, text, text_height=25):
        # Get image dimensions
        img_width, img_height = image.size

        # Create a new image with additional height for the text
        new_height = img_height + text_height
        new_img = Image.new("RGB", (img_width, new_height), "black")
        new_img.paste(image, (0, 0))

        # Draw text
        draw = ImageDraw.Draw(new_img)
        font = ImageFont.truetype(
            "arial.ttf", 10
        )  # Using default font; you can replace it with a custom font
        (left, top, right, bottom) = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = right - left, bottom - top
        text_position = (
            (img_width - text_width) // 2,
            img_height + (text_height // 4),
        )
        draw.text(text_position, text, font=font, fill="white")

        return new_img
    return add_text_below_image, create_gif_from_tifs


@app.cell(hide_code=True)
def __(create_gif_from_tifs, os):
    # get folder paths from inside brightfield
    def create_gifs():
        folder_path = "data/brightfield/"
        folder_names = os.listdir(folder_path)
        folder_paths = [
            os.path.join(folder_path, folder) for folder in folder_names
        ]
        print(folder_paths)
        output_gif_paths = [
            "data/" + folder_name + ".gif" for folder_name in folder_names
        ]
        print(output_gif_paths)
        for folder_path, output_path in zip(folder_paths, output_gif_paths):
            print(folder_path, output_path)
            create_gif_from_tifs(folder_path, output_path, duration=40)
        # create_gif_from_tifs(folder_paths[0], output_gif_paths[0], duration=50)


    create_gifs()
    return (create_gifs,)


@app.cell
def __(create_gif_from_tifs, os):
    # get folder paths from inside brightfield
    def create_gifs():
        folder_path = "data/brightfield/"
        folder_names = os.listdir(folder_path)
        folder_paths = [
            os.path.join(folder_path, folder) for folder in folder_names
        ]
        print(folder_paths)
        output_gif_paths = [
            "data/" + folder_name + ".gif" for folder_name in folder_names
        ]
        print(output_gif_paths)
        for folder_path, output_path in zip(folder_paths, output_gif_paths):
            print(folder_path, output_path)
            create_gif_from_tifs(folder_path, output_path, duration=40)
        # create_gif_from_tifs(folder_paths[0], output_gif_paths[0], duration=50)


    create_gifs()
    return (create_gifs,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Generate data + mask GIFs""")
    return


@app.cell
def __():
    from PIL import Image, ImageDraw, ImageFont
    import pandas as pd
    from preprocessing.dataframe_creator import (
        do_preprocessing,
        save_dataframe,
        load_dataframe_from_csv,
    )
    return (
        Image,
        ImageDraw,
        ImageFont,
        do_preprocessing,
        load_dataframe_from_csv,
        pd,
        save_dataframe,
    )


@app.cell
def __(load_dataframe_from_csv):
    df = load_dataframe_from_csv("data/filename_data.csv")
    df.head()
    return (df,)


@app.cell
def __(df):
    masks = df[df["has_mask"] == True][["sample_site", "z_position"]]
    # data_eq_mask = df[(df["has_mask"] == False) & (df['tag'].isin(masks))]
    data_eq_mask = df[
        (df["has_mask"] == False)
        & df[["sample_site", "z_position"]]
        .apply(tuple, axis=1)
        .isin(masks.apply(tuple, axis=1))
    ]
    # print(masks)
    data_eq_mask.head()
    return data_eq_mask, masks


@app.cell
def __(Image, ImageDraw, ImageFont, mo, pd):
    def create_composite_gifs(
        masks: pd.DataFrame,
        selected_data: pd.DataFrame,
        output_gif_path: str,
        duration=500,
    ):
        frames = []
        for mask_path, data_path in mo.status.progress_bar(list(zip(masks["path"], selected_data["path"]))):
            # Open images
            mask_image = Image.open(mask_path)
            data_image = Image.open(data_path)

            # Create a composite image
            composite_img = create_composite_image(
                data_image, mask_image, mask_path
            )
            frames.append(composite_img)

        # Save the frames as a GIF
        frames[0].save(
            output_gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
        )
        print(f"GIF created at {output_gif_path}")


    def create_composite_image(img1, img2, title, text_height=50):
        # Resize images to match each other (optional based on your needs)
        img2 = img2.resize(img1.size)

        # Combine the images side-by-side
        width, height = img1.size
        composite_width = width * 2
        composite_height = height + text_height

        composite_img = Image.new(
            "RGB", (composite_width, composite_height), "white"
        )
        composite_img.paste(img1, (0, 0))
        composite_img.paste(img2, (width, 0))

        # Draw the title text below the images
        draw = ImageDraw.Draw(composite_img)
        font = ImageFont.truetype(
            "arial.ttf", 10
        )  # Using default font; you can replace it with a custom font
        (left, top, right, bottom) = draw.textbbox((0, 0), title, font=font)
        text_width, text_height = right - left, bottom - top
        text_position = (
            (composite_width - text_width) // 2,
            composite_height + (text_height // 4),
        )
        draw.text(text_position, title, font=font, fill="white")
        
        return composite_img
    return create_composite_gifs, create_composite_image


@app.cell
def __(create_composite_gifs, data_eq_mask, df):
    output_folder = "data/brightfield2mask.gif"  # Output folder for generated GIFs
    create_composite_gifs(
        df[df["has_mask"] == True], data_eq_mask, output_folder, duration=500
    )
    return (output_folder,)


@app.cell
def __(mo):
    mo.md(r"""## Create and test dataframe""")
    return


@app.cell
def __():
    import pandas as pd
    import os
    import re
    return os, pd, re


@app.cell
def __(os, re):
    # Regular expression pattern to match the file format and extract parts
    # pattern = re.compile(
    #     r"(?P<name>[^_]+)_(?P<cell_type>[^_]+)_(?P<well_number>[^_]+)_(?P<location_number>[^_]+)_(?P<tag>s\d+z\d+c\d+)_(?P<data_type>ORG(?:_mask)?)\.(?P<file_format>tif{1,2})"
    # )
    pattern = re.compile(
        r"(?P<name>[^_]+)_(?P<cell_type>[^_]+)_well(?P<well_number>\d+)_(?P<location_number>[^_]+)_(?P<tag>s\d+z\d+c\d+)_(?P<data_type>ORG(?:_mask)?)\.(?P<file_format>tif{1,2})"
    )


    # List to store each row of extracted information
    data = []


    def process_files_in_directory(directory, has_subfolders=True):
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


    def append_tag_split(df):
        # Split the `tag` column into separate columns
        df["sample_site"] = df["tag"].str.extract(r"s(\d+)").astype(int)
        df["z_position"] = df["tag"].str.extract(r"z(\d+)").astype(int)
        df["channel"] = df["tag"].str.extract(r"c(\d+)").astype(int)
        return df


    def add_mask_indicator_column(df):
        df["has_mask"] = df["path"].str.contains("mask")
        return df
    return (
        add_mask_indicator_column,
        append_tag_split,
        data,
        pattern,
        process_files_in_directory,
    )


@app.cell
def __(
    add_mask_indicator_column,
    append_tag_split,
    data,
    pd,
    process_files_in_directory,
):
    # Process brightfield and masks directories
    process_files_in_directory("data/brightfield", has_subfolders=True)
    process_files_in_directory("data/masks", has_subfolders=False)

    # Create DataFrame from the list of dictionaries
    df = pd.DataFrame(data)
    df = append_tag_split(df)
    df = add_mask_indicator_column(df)

    # Drop the original `tag` column if no longer needed
    # df.drop(columns='tag', inplace=True)

    # Display the DataFrame
    print(df.head())
    return (df,)


@app.cell
def __(df, os):
    ms = os.listdir("data/masks")
    print(len(ms), ms[0])
    ms_filename = df.loc[df["has_mask"] == True, "filename"].to_list()
    print(len(ms_filename), ms_filename)
    differences = set(ms) ^ set(ms_filename)
    print(len(differences))
    print(differences)
    return differences, ms, ms_filename


@app.cell
def __(df):
    df["has_mask"].sum()
    return


@app.cell
def __():
    from preprocessing.dataframe_creator import (
        do_preprocessing,
        save_dataframe,
        load_dataframe_from_csv,
    )

    df2 = do_preprocessing()
    save_dataframe(df2, "./data/filename_data.csv")
    loaded_df = load_dataframe_from_csv("./data/filename_data.csv")
    df2.head()
    return (
        df2,
        do_preprocessing,
        load_dataframe_from_csv,
        loaded_df,
        save_dataframe,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Process data""")
    return


@app.cell
def __():
    from skimage.io import imread
    from skimage.transform import resize
    import imagecodecs
    return imagecodecs, imread, resize


@app.cell(hide_code=True)
def __(mo):
    mo.md("""Get number of training and testing data""")
    return


@app.cell
def __(glob):
    training_paths = [
        path.replace("\\", "/") for path in glob.glob("data/brightfield/*/*.tif")
    ]
    print(f"Total: {len(training_paths)} images")
    return (training_paths,)


@app.cell
def __(glob):
    test_paths = [path.replace("\\", "/") for path in glob.glob("data/masks/*")]
    print(f"Total: {len(test_paths)} masks")
    return (test_paths,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Check if all images have the same size""")
    return


@app.cell
def __(imread, test_paths, training_paths):
    def get_size(path):
        return imread(path, as_gray=True).shape


    training_sizes = [get_size(path) for path in training_paths]
    print(set(training_sizes))
    test_sizes = [get_size(path) for path in test_paths]
    print(set(test_sizes))
    return get_size, test_sizes, training_sizes


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Result: all images have 1024x1024 size""")
    return


@app.cell
def __(mo):
    mo.md(r""" """)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Test data should be: well1
        Training and validation: well2-7

        """
    )
    return


if __name__ == "__main__":
    app.run()
