from datasets import load_dataset
import pandas as pd
from typing import Dict, List

dataset = load_dataset("Anthropic/hh-rlhf")


# Load the red teaming subset
dataset = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts")

df = pd.DataFrame(dataset["train"])

# Most negative scored transcripts are typically the most unhinged.
df.sort_values(by="min_harmlessness_score_transcript", inplace=True)

transcripts = df["transcript"].tolist()
transcripts = [
    t.replace("Human: ", "").replace("Assistant: ", "").split("\n\n")
    for t in transcripts
]

# Rylan hand-chose these particular indices.
# The notation is: (index, human prompt, )
tuples_of_harmful_generations = [
    (0, (1,), (2, 4)),
    (0, (1,), (2, 6)),
    (0, (1,), (8,)),
    (1, (1,), (2,)),
    (1, (1,), (6,)),
    (4, (1, 3), (6, 7)),
    (5, (1,), (8, 9)),
    (5, (1,), (8, 10)),
    (5, (1,), (8, 15)),
    (6, (1,), (15,)),
    (6, (1,), (20,)),
    (7, (1,), (12, 13)),
    (7, (1,), (13,)),
    (8, (1, 3), (9,)),
    (9, (1,), (2, 5, 6)),
    (9, (1,), (4, 5, 7)),
    (10, (1,), (6,)),
    (11, (1,), (8, 10)),
    (12, (1,), (4, 6)),
    (13, (1,), (2, 6)),
    (13, (1,), (4, 6)),
    (14, (1, 3, 5), (7, 9)),
    (17, (1,), (2,)),
    (
        17,
        (1,),
        (6, 7),
    ),
    (
        18,
        (1, 3),
        (6,),
    ),
    (
        19,
        (1,),
        (
            4,
            6,
        ),
    ),
    (
        19,
        (1,),
        (4, 8),
    ),
    (21, (1,), (6, 7)),
    (23, (1, 3), (4, 7)),
    (23, (1, 3), (4, 9)),
    (24, (1,), (8,)),
    (27, (1,), (4, 1)),
    (35, (1,), (8,)),
    (36, (1,), (6,)),
    (39, (1,), (4, 5)),
    (64, (1,), (4,)),
    (66, (1,), (8, 9)),
    (71, (1,), (6,)),
    (73, (1,), (2,)),
    (77, (1,), (6,)),
    (101, (1,), (4, 14, 15)),
    (118, (1, 2), (12,)),
    (132, (1,), (2, 4)),
    (138, (1,), (6, 13)),
    (155, (1,), (4,)),
    (160, (1,), (4, 6, 7)),
    (167, (1,), (4, 6)),
    (168, (1,), (6, 7)),
    (174, (1, 3), (4, 5)),
    (177, (1,), (4, 5, 6)),
    (186, (1,), (2,)),
    (187, (1, 5), (7, 8)),
    (189, (1,), (5,)),
    (191, (1,), (4,)),
    (192, (3,), (4,)),
    (195, (1,), (4,)),
    (198, (1,), (2,)),
    (199, (1,), (6,)),
    (201, (1,), (4, 5)),
    (208, (1,), (2, 3)),
    (209, (1,), (2,)),
    (210, (1,), (2,)),
    (210, (1,), (8, 9)),
    (212, (1,), (2,)),
    (215, (3,), (4,)),
    (215, (5,), (6, 8)),
    (217, (1,), (8,)),
    (221, (1,), (4,)),
    (222, (3,), (4,)),
    (225, (1,), (2,)),
    (227, (1,), (4, 5)),
    (233, (1,), (6,)),
    (235, (1,), (2,)),
    (243, (1,), (2,)),
    (244, (1,), (2, 3, 4)),
    (245, (1,), (4,)),
    (249, (5,), (6, 7)),
    (255, (1, 5), (6,)),
    (257, (1,), (4, 5)),
    (261, (3,), (4,)),
    (261, (5,), (9,)),
    (270, (1,), (2,)),
    (276, (1,), (4,)),
    (278, (1,), (10,)),
    (
        279,
        (1,),
        (
            6,
            8,
        ),
    ),
    (281, (1,), (8,)),
    (283, (1, 3), (6,)),
    (292, (5,), (6,)),
    (299, (5,), (6, 7)),
    (302, (1,), (2,)),
    (303, (1,), (6,)),
    (307, (1,), (4, 5)),
    (310, (1, 7), (8,)),
    (310, (1, 7), (12,)),
    (313, (1,), (2,)),
    (317, (1,), (2,)),
    (324, (1,), (4, 5)),
    (324, (1,), (4, 6)),
    (325, (1,), (7, 8)),
    (334, (1,), (9,)),
    (354, (1,), (6, 7)),
    (357, (1,), (9,)),
    (374, (1,), (8,)),
    (377, (1,), (2,)),
    (380, (1,), (13,)),
    (383, (1, 7), (8,)),
]

rylan_extracted_transcripts: List[Dict[str, str]] = []
max_target_string_length = 100
for tuples_of_harmful_generation in tuples_of_harmful_generations:
    (
        transcript_idx,
        human_prompt_indices,
        model_generation_indices,
    ) = tuples_of_harmful_generation
    rylan_extracted_transcripts.append(
        {
            "prompt": " ".join(
                [transcripts[transcript_idx][idx] for idx in human_prompt_indices]
            ),
            "target": " ".join(
                [transcripts[transcript_idx][idx] for idx in model_generation_indices]
            )[:max_target_string_length],
        }
    )

rylan_extracted_df = pd.DataFrame(rylan_extracted_transcripts)
rylan_extracted_df.to_csv(
    "prompts_and_targets/anthropic_hhh/red_team_attempts.csv", index=False
)

# print(1)
