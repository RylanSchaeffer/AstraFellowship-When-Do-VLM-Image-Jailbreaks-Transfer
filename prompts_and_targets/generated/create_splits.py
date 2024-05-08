# %%
import pandas as pd
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(f"{current_dir}/generated_dataset.csv")

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Truncate the "target" column to 150 characters
df["target"] = df["target"].apply(lambda x: x[:250])

train_size = int(len(df) * 0.8)

train_df = df[:train_size]
eval_df = df[train_size:]

train_df.to_csv(f"{current_dir}/train.csv", index=False)
eval_df.to_csv(f"{current_dir}/eval.csv", index=False)
