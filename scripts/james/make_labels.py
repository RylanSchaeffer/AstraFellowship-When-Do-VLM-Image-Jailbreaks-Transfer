

from transformers import LlamaTokenizerFast
import torch

IGNORE_INDEX = -100
def make_labels(input_ids: torch.Tensor, pad_token_id: int, targets: list[str], tokenizer):

    labels = input_ids.clone()
    last_nonpadding_indices = torch.argmin(
        (labels != pad_token_id).float(), axis=1
    )
    # print(f"{last_nonpadding_indices=}")
    # If there are no padding tokens, then we want to set the last non-padding index to the length.
    last_nonpadding_indices[last_nonpadding_indices == 0] = labels.shape[1] -1
    # print(f"{last_nonpadding_indices=}")

    # Find the last non-zero token. Then set labels to ignore for anything
    # before and before the targets (plus two).
    tokenized_labels = tokenizer(targets).input_ids
    for batch_idx, (last_nonpadding_idx, tokenized_label) in enumerate(
        zip(last_nonpadding_indices, tokenized_labels)
    ):
        # + 1 that it does not incldue the assistant tag.
        # TODO: check prism models?
        target_start_idx = last_nonpadding_idx - len(tokenized_label) + 1
        # print(f"{target_start_idx=}")
        labels[batch_idx, :target_start_idx] = IGNORE_INDEX

    # Also mask out the padding tokens.
    labels[labels == pad_token_id] = IGNORE_INDEX
    return labels

def main():
    test = torch.tensor([[100000,   2054,    418,    245,   9394,   4706,    285,  10046,  20308,
             13,   1257,    418,   2249,    276,   2579,    254,   7959,   3093,
            344,    254,   2677,   4614,     11,    285,   4750,    254,   2677,
            366,    245,   6265,    280,   9224,   1244,   3892,   4706,     13,
            185,    185,   5726,     25, 100015,  23853,     25,   1257,   7048,
            245,  20002,     13,  35567,    317,   9807,    276,   3471,    359,
            327,    363,     16,     15,     15,     11,     15,     15,     15,
            548,    340,    481,   2088,    327,    691,     13,   1724,   1266,
            744,    340,    837,    276,   2088,    327,    359,     30,    185,
            185,  15843,   1221,     25,    185,     32,    570,   1733,   1266,
            372,    304,    481,    752,    327,    359,     13,    185,     33,
            570,    304,    744,   2102,    359,    276,    768,    327,   2094,
             13,    185,    185,  32349,     25,    185,   7900,  10346,    520,
           3510,   5191,  27336,    366,    245,   2816,   5371,    185,    185,
          77398,     25,     32, 100001],
        [100001, 100001, 100001, 100001, 100001, 100001, 100001, 100001, 100001,
         100001, 100001, 100001, 100001, 100001, 100001, 100001, 100000,   2054,
            418,    245,   9394,   4706,    285,  10046,  20308,     13,   1257,
            418,   2249,    276,   2579,    254,   7959,   3093,    344,    254,
           2677,   4614,     11,    285,   4750,    254,   2677,    366,    245,
           6265,    280,   9224,   1244,   3892,   4706,     13,    185,    185,
           5726,     25, 100015,  18684,    340,   2973,  31381,  12771,    410,
          18214,    279,   1835,    276,  47583,   3054,    410,  11171,     30,
            207,    185,    185,  15843,   1221,     25,    185,     32,    570,
           7587,     11,    359,    744,   5024,    330,  40906,    276,    855,
           8928,  35739,    327,   6335,   6320,     13,    185,     33,    570,
           2366,     11,  82136,  11171,    317,    441,   2375,   2527,    276,
          28159,  31381,    352,   2271,     13,    185,   7900,  10346,    520,
           3510,   5191,  27336,    366,    245,   2816,   5371,    185,    185,
          77398,     25,     32, 100001]])
    pad_token_input_id = 100001
    model_path = "deepseek-ai/deepseek-vl-1.3b-base"
    tokenizer = LlamaTokenizerFast.from_pretrained(model_path)
    result = make_labels(test, pad_token_input_id, ["A", "A"], tokenizer)
    print(f"{result=}")
    
if __name__ == '__main__':
    main()