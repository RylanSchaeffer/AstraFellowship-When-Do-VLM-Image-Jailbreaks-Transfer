IGNORE_INDEX = -100
import torch


def make_labels(
    input_ids: torch.Tensor, pad_token_id: int, targets: list[str], tokenizer
):
    labels = input_ids.clone()
    last_nonpadding_indices = torch.argmin((labels != pad_token_id).float(), axis=1)
    # print(f"{last_nonpadding_indices=}")
    # If there are no padding tokens, then we want to set the last non-padding index to the length.
    last_nonpadding_indices[last_nonpadding_indices == 0] = (
        labels.shape[1] - 1
    )  # Minus one!!
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
