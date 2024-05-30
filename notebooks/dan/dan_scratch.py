# %%
# import os


# def rename_files(directory):
#     for filename in os.listdir(directory):
#         if filename.endswith(".csv"):
#             parts = filename.split("_")
#             new_name = (
#                 parts[2].lower().replace(" ", "_") + ".csv"
#             )  # create the new filename
#             os.rename(
#                 os.path.join(directory, filename), os.path.join(directory, new_name)
#             )
#             print(f"Renamed {filename} to {new_name}")


# def move_files_to_subdirs(directory):
#     for filename in os.listdir(directory):
#         if filename.endswith(".csv"):
#             subdir = os.path.join(
#                 directory, filename[:-4]
#             )  # use the file name without .csv for the subdir name

#             # Create subdirectory if it doesn't exist
#             if not os.path.exists(subdir):
#                 os.makedirs(subdir)
#                 print(f"Created directory: {subdir}")

#             # Move the file to the new directory
#             os.rename(os.path.join(directory, filename), os.path.join(subdir, filename))
#             print(f"Moved {filename} to {subdir}/{filename}")


# # Specify the directory where your files are located
# directory = "/Users/dan/generated_topics/"
# # rename_files(directory)
# move_files_to_subdirs(directory)


# # %%
# import os
# import pandas as pd


# def concat_csv_files(directory_path):
#     # List all CSV files in the directory
#     csv_files = [f for f in os.listdir(directory_path) if f.endswith(".csv")]

#     # Read and concatenate all CSV files into a single DataFrame
#     df = pd.concat(
#         [pd.read_csv(os.path.join(directory_path, file)) for file in csv_files],
#         ignore_index=True,
#     )

#     return df


# # Specify the directory containing the CSV files
# directory_path = "/Users/dan/generated_fixed"

# # Concatenate all CSV files in the directory
# concatenated_df = concat_csv_files(directory_path)

# # Optionally, save the concatenated DataFrame to a new CSV file
# concatenated_df.to_csv("/Users/dan/generated_dataset.csv", index=False)


# # %%
# #
# # Load the CSV file
# # base_dir = os.path.dirname(os.path.abspath(__file__))
# base_dir = "/Users/dan/projects/vlm_jailbreak/prompts_and_targets/generated"
# df = pd.read_csv(f"{base_dir}/generated_dataset.csv")

# # Print the number of rows
# print("Number of rows:", len(df))

# %%

import random
from prompts_and_targets.generated.generated_dataset import (
    GeneratedPromptResponseDataset,
)
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import pandas as pd
import os

generated_dataset = GeneratedPromptResponseDataset.from_file(
    "./prompts_and_targets/generated/generated_dataset.csv"
)
train, eval = generated_dataset.create_splits(portion=1, split=0.8, target_len=250)
sample = random.sample(train.items, 10)
for i in sample:
    print(i.prompt)
df = pd.DataFrame([asdict(item) for item in train.items])
sample = df.sample(10)
# prompts_and_targets_path = os.path.join(
#     "prompts_and_targets", "anthropic_hhh", "train.csv"
# )
# df = pd.read_csv(prompts_and_targets_path)

# df["prompt_length"] = df["prompt"].str.len()

# # Create a histogram of the prompt lengths
# plt.figure(figsize=(10, 6))
# plt.hist(df["prompt_length"], bins=200, edgecolor="black")
# plt.xlabel("Prompt Length")
# plt.ylabel("Frequency")
# plt.title("Distribution of Prompt Lengths")
# plt.ylim(0, 1000)
# plt.grid(True)
# plt.show()

# from statsmodels.distributions.empirical_distribution import ECDF

# # Assuming your DataFrame is named 'df' and has a column named 'prompt'

# # Calculate the length of each prompt
# prompt_lengths = df["prompt"].str.len()

# # Create the ECDF
# ecdf = ECDF(prompt_lengths)
# print(f"The cumulative probability at prompt length {250} is {ecdf(250)}")

# # Create the ECDF plot
# plt.figure(figsize=(10, 6))
# plt.plot(ecdf.x, ecdf.y, marker=".", linestyle="none")
# plt.xlabel("Prompt Length")
# plt.ylabel("Cumulative Probability")
# plt.title(
#     "Ant hhh - Empirical Cumulative Distribution Function (ECDF) of Prompt Lengths"
# )
# plt.grid(True)
# plt.show()
# # topic_counts = train_df["topic_name"].value_counts()

# # Print the unique values and their counts
# # for topic, count in topic_counts.items():
# #     print(f"Topic: {topic}, Count: {count}")

# # print(len(train_df))
