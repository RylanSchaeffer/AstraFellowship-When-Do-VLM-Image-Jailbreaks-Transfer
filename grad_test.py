
import torch

# make a tensor
x = torch.tensor([1, 2, 3, 4, 5])
# check grad
print(f"x.requires_grad: {x.requires_grad=}")