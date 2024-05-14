"""
upload_prismatic_vlms.py

Utility script for uploading VLA checkpoints to the HuggingFace Hub, under the `TRI-ML/prismatic-vlms` model repository.
Makes it easy to share/load arbitrary VLM checkpoints via `hf_hub_download` (with built-in caching logic). Note that
this script loads the (full precision/FP32 checkpoints, and converts to BF16, prior to uploading).

Preliminaries:
    - Install the Rust-based HF-Transfer Extension --> `pip install --upgrade huggingface_hub[hf_transfer]`
        - Enable: `export HF_HUB_ENABLE_HF_TRANSFER=1`
    - Login via the HuggingFace CLI --> `huggingface-cli login`. Make sure to provide a token with WRITE permissions.
    - Verify that `openvla` is in "orgs" --> `huggingface-cli whoami`

Run with `python scripts/hf-hub/upload_prismatic_vlms.py \
    --model_id "prism-dinosiglip-224px+7b" \
    --run_dir "/mnt/fsx/x-prismatic-vlms/runs/llava-lvis4v-lrv+prism-dinosiglip-224px+7b+stage-finetune+x7"`
"""

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import draccus
import torch


@dataclass
class UploadConfig:
    # fmt: off
    model_id: str                               # Short Model Identifier to use in HF Hub Uploads
    run_dir: Union[str, Path]                   # Absolute Path to Top-Level Run Directory (contains `config.yaml`)

    conversion_dir = Path(                      # Parent Directory to Store BF16 Converted Checkpoints
        "/mnt/fsx/x-prismatic-vlms/release/"
    )

    hub_repo: str = "TRI-ML/prismatic-vlms"     # HF Hub Repository

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)

        # Validate
        assert self.run_dir.exists() and (self.run_dir / "config.yaml").exists(), "Missing config.yaml in `run_dir`!"

    # fmt: on


@draccus.wrap()
def upload(cfg: UploadConfig) -> None:
    print(
        f"[*] Uploading from `{cfg.run_dir}` to HF Hub :: `{cfg.hub_repo}/{cfg.model_id}`"
    )

    # Set Conversion Path
    convert_path = cfg.conversion_dir / cfg.model_id
    os.makedirs(convert_path / "checkpoints", exist_ok=True)

    # Copy Top-Level Config & Metrics Files to `convert_path`
    print(f"[*] Copying Top-Level Config & Metric Files to `{convert_path}`")
    for fn in ["config.json", "config.yaml", "run-metrics.jsonl"]:
        shutil.copy(cfg.run_dir / fn, convert_path / fn)

    # Convert `latest-checkpoint.pt` to BF16
    print("[*] Converting VLM Checkpoint to BF16")
    model_state_dict = torch.load(
        cfg.run_dir / "checkpoints" / "latest-checkpoint.pt", map_location="cpu"
    )
    for mkey in model_state_dict["model"]:
        for pkey in model_state_dict["model"][mkey].keys():
            assert (
                model_state_dict["model"][mkey][pkey].dtype == torch.float32
            ), "Expected Tensor in FP32!"
            model_state_dict["model"][mkey][pkey] = model_state_dict["model"][mkey][
                pkey
            ].to(torch.bfloat16)

    # Save BF16 Checkpoint
    print(f"[*] Saving BF16 Checkpoint to `{convert_path}`")
    torch.save(model_state_dict, convert_path / "checkpoints" / "latest-checkpoint.pt")

    # Upload!
    print(f"[*] Uploading VLM `{cfg.model_id}` (will take a while...)")
    subprocess.run(
        f"HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli upload {cfg.hub_repo} {convert_path!s} {cfg.model_id}/",
        shell=True,
        check=True,
    )

    # Done
    print(
        f"\n[*] Done =>> Check https://huggingface.co/{cfg.hub_repo}/tree/main/{cfg.model_id}"
    )


if __name__ == "__main__":
    upload()
