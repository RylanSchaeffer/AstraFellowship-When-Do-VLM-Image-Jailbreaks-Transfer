import getpass
import numpy as np
import random
import torch


def retrieve_wandb_username() -> str:
    system_username = getpass.getuser()
    if system_username == "rschaef":
        wandb_username = "rylan"
    else:
        raise ValueError(f"Unknown system username: {system_username}")
    return wandb_username


def set_seed(seed=1):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
