# %%
import os


def rename_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            parts = filename.split("_")
            new_name = (
                parts[2].lower().replace(" ", "_") + ".csv"
            )  # create the new filename
            os.rename(
                os.path.join(directory, filename), os.path.join(directory, new_name)
            )
            print(f"Renamed {filename} to {new_name}")


def move_files_to_subdirs(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            subdir = os.path.join(
                directory, filename[:-4]
            )  # use the file name without .csv for the subdir name

            # Create subdirectory if it doesn't exist
            if not os.path.exists(subdir):
                os.makedirs(subdir)
                print(f"Created directory: {subdir}")

            # Move the file to the new directory
            os.rename(os.path.join(directory, filename), os.path.join(subdir, filename))
            print(f"Moved {filename} to {subdir}/{filename}")


# Specify the directory where your files are located
directory = "/Users/dan/generated_topics/"
# rename_files(directory)
move_files_to_subdirs(directory)
