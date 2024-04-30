from accelerate import Accelerator
import ast
import getpass

import pydantic
import joblib
import numpy as np
import os
import pandas as pd
from PIL import Image
import random
import torch
import torch.distributed
import torch.utils.data
import torchvision.transforms.v2
from typing import Any, Dict, List, Tuple
import wandb

from src.data import VLMEnsembleTextDataset, VLMEnsembleTextDataModule
from src.models.ensemble import VLMEnsemble
from src.image_handling import get_list_image
from src.openai_utils.client import encode_image


def calc_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return torch.distributed.get_rank()


def create_initial_image(image_kwargs: Dict[str, Any], seed: int = 0) -> torch.Tensor:
    if image_kwargs["image_initialization"] == "NIPS17":
        image = get_list_image("old/how_robust_is_bard/src/dataset/NIPS17")
        # resizer = transforms.Resize((224, 224))
        # images = torch.stack(
        #     [resizer(i).unsqueeze(0).to(torch.float16) for i in images]
        # )
        # # Only use one image for one attack.
        # images: torch.Tensor = images[image_kwargs["datum_index"]].unsqueeze(0)
        raise NotImplementedError
    elif image_kwargs["image_initialization"] == "random":
        image_size = image_kwargs["image_size"]
        image: torch.Tensor = torch.rand((1, 3, image_size, image_size))
    elif image_kwargs["image_initialization"] == "trina":
        image_path = f"images/trina/{str(seed).zfill(3)}.jpg"
        pil_image = Image.open(image_path, mode="r")
        width, height = pil_image.size
        max_dim = max(width, height)
        pad_width = (max_dim - width) // 2
        pad_height = (max_dim - height) // 2
        transform_pil_image = torchvision.transforms.v2.Compose(
            [
                torchvision.transforms.v2.Pad(
                    (pad_width, pad_height, pad_width, pad_height), fill=0
                ),
                torchvision.transforms.v2.Resize(
                    (image_kwargs["image_size"], image_kwargs["image_size"])
                ),
                torchvision.transforms.v2.ToTensor(),  # This divides by 255.
            ]
        )
        image: torch.Tensor = transform_pil_image(pil_image).unsqueeze(0)
    else:
        raise ValueError(
            "Invalid image_initialization: {}".format(
                image_kwargs["image_initialization_str"]
            )
        )
    assert len(image.shape) == 4
    return image


def instantiate_vlm_ensemble(
    model_strs: List[str],
    model_generation_kwargs: Dict[str, Dict[str, Any]],
    accelerator: Accelerator,
) -> VLMEnsemble:
    # TODO: This function is probably overengineered and should be deleted.
    vlm_ensemble = VLMEnsemble(
        model_strs=model_strs,
        model_generation_kwargs=model_generation_kwargs,
        accelerator=accelerator,
    )
    vlm_ensemble = accelerator.prepare([vlm_ensemble])[0]
    return vlm_ensemble


def is_dist_avail_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def load_jailbreak_dicts_list(
    wandb_run_id: str = None,
    wandb_sweep_id: str = None,
    data_dir_path: str = "eval_data",
    refresh: bool = False,
) -> List[Dict[str, Any]]:
    os.makedirs(data_dir_path, exist_ok=True)
    runs_jailbreak_dict_list_path = os.path.join(
        data_dir_path,
        f"runs_jailbreak_dict_list_sweep={wandb_run_id}.joblib",
    )
    if refresh or not os.path.exists(runs_jailbreak_dict_list_path):
        print("Downloading jailbreak images...")

        api = wandb.Api()
        if wandb_sweep_id is None and wandb_run_id is not None:
            run = api.run(f"universal-vlm-jailbreak/{wandb_run_id}")
            runs = [run]
        elif wandb_sweep_id is not None and wandb_run_id is None:
            sweep = api.sweep(f"universal-vlm-jailbreak/{wandb_run_id}")
            runs = list(sweep.runs)
        else:
            raise ValueError(
                "Invalid wandb_sweep_id and wandb_run_id: "
                f"{wandb_sweep_id}, {wandb_run_id}"
            )
        runs_jailbreak_dict_list = []
        for run in runs:
            for file in run.files():
                file_name = str(file.name)
                if not file_name.endswith(".png"):
                    continue
                file_dir_path = os.path.join(data_dir_path, run.id)
                os.makedirs(file_dir_path, exist_ok=True)
                file.download(root=file_dir_path, replace=True)
                # Example:
                #   'eval_data/sweep=7v3u4uq5/dz2maypg/media/images/jailbreak_image_step=500_0_6bff027c89aa794cfb3b.png'
                # becomes
                #   500
                optimizer_step_counter = int(file_name.split("_")[2][5:])
                file_path = os.path.join(file_dir_path, file_name)
                runs_jailbreak_dict_list.append(
                    {
                        "file_path": file_path,
                        "wandb_run_id": run.id,
                        "optimizer_step_counter": optimizer_step_counter,
                        "attack_models_str": run.config["models_to_attack"],
                    }
                )

                print(
                    "Downloaded jailbreak image for run: ",
                    run.id,
                    " at optimizer step: ",
                    optimizer_step_counter,
                )

        # Sort runs_jailbreak_dict_list based on wandb_run_id and then n_gradient_steps.
        runs_jailbreak_dict_list = sorted(
            runs_jailbreak_dict_list,
            key=lambda x: (x["wandb_run_id"], x["optimizer_step_counter"]),
        )

        joblib.dump(
            value=runs_jailbreak_dict_list,
            filename=runs_jailbreak_dict_list_path,
        )

        print("Saved runs_jailbreak_dict_list to: ", runs_jailbreak_dict_list_path)

    else:
        runs_jailbreak_dict_list = joblib.load(runs_jailbreak_dict_list_path)

        print("Loaded runs_jailbreak_dict_list from: ", runs_jailbreak_dict_list_path)

    return runs_jailbreak_dict_list


class JailbreakData(pydantic.BaseModel):
    file_path: str
    wandb_run_id: str
    optimizer_step_counter: int
    attack_models_str: str
    image_base_64: str
    

def load_jailbreak_list(
    wandb_run_id: str | None = None,
    wandb_sweep_id: str | None= None,
    data_dir_path: str = "eval_data",
    refresh: bool = False,
) -> List[JailbreakData]:
    os.makedirs(data_dir_path, exist_ok=True)
    runs_jailbreak_dict_list_path = os.path.join(
        data_dir_path,
        f"runs_jailbreak_dict_list_sweep={wandb_run_id}.joblib",
    )
    if refresh or not os.path.exists(runs_jailbreak_dict_list_path):
        print("Downloading jailbreak images...")

        api = wandb.Api()
        if wandb_sweep_id is None and wandb_run_id is not None:
            run = api.run(f"universal-vlm-jailbreak/{wandb_run_id}")
            runs = [run]
        elif wandb_sweep_id is not None and wandb_run_id is None:
            sweep = api.sweep(f"universal-vlm-jailbreak/{wandb_run_id}")
            runs = list(sweep.runs)
        else:
            raise ValueError(
                "Invalid wandb_sweep_id and wandb_run_id: "
                f"{wandb_sweep_id}, {wandb_run_id}"
            )
        runs_jailbreak_dict_list = []
        for run in runs:
            for file in run.files():
                file_name = str(file.name)
                if not file_name.endswith(".png"):
                    continue
                file_dir_path = os.path.join(data_dir_path, run.id)
                os.makedirs(file_dir_path, exist_ok=True)
                file.download(root=file_dir_path, replace=True)
                # Example:
                #   'eval_data/sweep=7v3u4uq5/dz2maypg/media/images/jailbreak_image_step=500_0_6bff027c89aa794cfb3b.png'
                # becomes
                #   500
                optimizer_step_counter = int(file_name.split("_")[2][5:])
                file_path = os.path.join(file_dir_path, file_name)
                runs_jailbreak_dict_list.append(
                    JailbreakData(
                        file_path=file_path,
                        wandb_run_id=run.id,
                        optimizer_step_counter=optimizer_step_counter,
                        attack_models_str=run.config["models_to_attack"],
                        image_base_64=encode_image(file_path),
                    )
                )

                print(
                    "Downloaded jailbreak image for run: ",
                    run.id,
                    " at optimizer step: ",
                    optimizer_step_counter,
                )

        # Sort runs_jailbreak_dict_list based on wandb_run_id and then n_gradient_steps.
        runs_jailbreak_dict_list = sorted(
            runs_jailbreak_dict_list,
            key=lambda x: (x.wandb_run_id, x.optimizer_step_counter),
        )

        joblib.dump(
            value=runs_jailbreak_dict_list,
            filename=runs_jailbreak_dict_list_path,
        )

        print("Saved runs_jailbreak_dict_list to: ", runs_jailbreak_dict_list_path)

    else:
        runs_jailbreak_dict_list = joblib.load(runs_jailbreak_dict_list_path)

        print("Loaded runs_jailbreak_dict_list from: ", runs_jailbreak_dict_list_path)

    return runs_jailbreak_dict_list




def retrieve_wandb_username() -> str:
    # system_username = getpass.getuser()
    # if system_username == "rschaef":
    #     wandb_username = "rylan"
    # else:
    #     raise ValueError(f"Unknown W&B username: {system_username}")
    import wandb

    api = wandb.Api()
    wandb_username = api.viewer.username
    return wandb_username


def set_seed(seed=1):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        import torch.backends.cudnn as cudnn

        cudnn.benchmark = False
        cudnn.deterministic = True
    except ImportError:
        pass
