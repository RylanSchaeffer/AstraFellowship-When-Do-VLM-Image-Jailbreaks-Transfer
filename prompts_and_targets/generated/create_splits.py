# %%
import pandas as pd
import numpy as np
import os

# current_dir = os.path.dirname(os.path.abspath(__file__))
# df = pd.read_csv(f"{current_dir}/generated_dataset.csv")

# df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# # Truncate the "target" column to 150 characters
# df["target"] = df["target"].apply(lambda x: x[:150])

# train_size = int(len(df) * 0.8)

# train_df = df[:train_size]
# eval_df = df[train_size:]

# train_df.to_csv(f"{current_dir}/train.csv", index=False)
# eval_df.to_csv(f"{current_dir}/eval.csv", index=False)


def process_csv_file(file_path, output_dir):
    # Read the dataset
    df = pd.read_csv(file_path)

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df["target"] = df["target"].apply(lambda x: x[:250])

    # Split the data into train and eval
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size]
    eval_df = df[train_size:]

    # Save the datasets
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    eval_df.to_csv(os.path.join(output_dir, "eval.csv"), index=False)


def delete_existing_files(output_dir):
    # Delete existing train and eval files if they exist
    for filename in ["train.csv", "eval.csv"]:
        file_path = os.path.join(output_dir, filename)
        if os.path.exists(file_path):
            os.remove(file_path)


def process_all_subdirectories(base_dir):
    # List all subdirectories in the base directory
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)

        # Check if the path is a directory
        if os.path.isdir(subdir_path):
            # First, delete any existing train or eval files
            delete_existing_files(subdir_path)

            # Assume there is one CSV file per subdir
            csv_file = next(
                (
                    f
                    for f in os.listdir(subdir_path)
                    if f.endswith(".csv") and "train" not in f and "eval" not in f
                ),
                None,
            )
            if csv_file:
                csv_file_path = os.path.join(subdir_path, csv_file)
                process_csv_file(csv_file_path, subdir_path)
            else:
                print(f"No CSV file found in {subdir_path}")
        else:
            print(f"{subdir_path} is not a directory")


# Example usage:
base_dir = os.path.dirname(os.path.abspath(__file__))
process_all_subdirectories(f"{base_dir}/generated_topics")

# %%
