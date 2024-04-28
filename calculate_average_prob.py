import numpy as np
import torch

# Calculate the probability from the loss
loss = 1.0
probability = np.exp(-loss)
print(probability)

# proba = 0.5
# log = np.log(proba)
# print(-log)
# probas =torch.Tensor([0.1, 0.2, 0.7])
# log_scale = probas.log()
# print(log_scale)
# print(log_scale.softmax(-1))