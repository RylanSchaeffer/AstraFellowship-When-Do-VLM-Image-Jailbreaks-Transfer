import pandas as pd

# read run.csv
df = pd.read_csv("scripts/run.csv")


def bool_strs_to_list_bools(bool_strs):
    return [bool_str == "true" for bool_str in bool_strs.split(",")]


def calculate_accuracy(bool_strs):
    list_bools = bool_strs_to_list_bools(bool_strs=bool_strs)
    return sum(list_bools) / len(list_bools)


df["accuracy"] = df["matches_target"].apply(calculate_accuracy)
print(df["accuracy"])
