# %%

import re
import json
import wandb
from src.models.evaluators import Claude3OpusEvaluator

api = wandb.Api()

# entity = "rylan"
entity = "danvalentine256"
project = "universal-vlm-jailbreak-eval"
# sweep_id = "cubnlte5"
sweep_id = "dvv5o3q6"

sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
runs = [r for r in sweep.runs if r.state in ["finished"]]
print("Finished runs:", runs)

claude = Claude3OpusEvaluator(20)
for idx, run in enumerate(runs):
    print(f"{idx}/{len(runs)} -> Run ID: {run.id}, Name: {run.name}")
    # need to resume the run to upload files to it - context manager will close it again when done
    with wandb.init(
        project=project, entity=entity, id=run.id, resume="must"
    ) as resumed_run:
        files = run.files()
        table_file_pattern = r"^media/table/generations.*\.table\.json$"
        files = [f for f in files if re.match(table_file_pattern, f.name)]
        for idx, file in enumerate(files):
            print(f"{idx}/{len(files)} -> File: {file.name}")
            with file.download(replace=True, exist_ok=True) as old_file:
                table = json.load(old_file)
                prompt_idx = table["columns"].index("prompt")
                generated_idx = table["columns"].index("generated")
                claude_idx = table["columns"].index("claude3opus")
                claude_errors = [
                    row[claude_idx]
                    for row in table["data"]
                    if "Error" in row[claude_idx]
                ]
                # claude_errors = ["Error"]  # for testing without error files
                if len(claude_errors) == 0:
                    # just copy the file and upload it
                    print(f"No errors in {file.name}")
                    new_table = wandb.Table(
                        columns=table["columns"], data=table["data"]
                    )
                else:
                    print(f"Re-running Claude for {file.name}")
                    new_data = []
                    prompts = [row[prompt_idx] for row in table["data"]]
                    generations = [row[generated_idx] for row in table["data"]]
                    evaluations = claude.evaluate(prompts, generations)
                    assert len(evaluations) == len(generations)
                    for evaluation, row in zip(evaluations, table["data"]):
                        row[claude_idx] = evaluation
                        new_data.append(row)

                    new_table = wandb.Table(columns=table["columns"], data=new_data)

                filename = file.name.split("media/table/")[1].split(".table.json")[0]
                new_table_name = f"repaired_{filename}"
                print(new_table_name)
                resumed_run.log({new_table_name: new_table})
