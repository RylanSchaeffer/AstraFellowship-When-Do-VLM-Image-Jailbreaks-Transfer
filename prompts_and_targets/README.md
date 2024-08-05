# Prompts and Targets

## Data Sourcing

- [AdvBench](https://github.com/llm-attacks/llm-attacks/tree/main) `advbench`
  - `harmful_behaviors.csv`: Copied from https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv on 2024/02/19
- [Anthropic HHH]() TODO


Note! I needed `datasets==2.8.0` in order to download the Anthropic dataset. Otherwise it threw this funky error:

https://huggingface.co/datasets/Anthropic/hh-rlhf/discussions/14

## Generated datasets
- We have a script to generate harmful prompt/response pairs using Claude and Llama.
- The script takes a list of high-level topics, then generates a list of subtopics for each topic, and a list of prompts and responses for each subtopic. Responses are filtered to keep only harmful ones.
- Generate a dataset from the command line:  `python -m prompts_and_targets.generated.generate_dataset [...args]`
    - Arguments:
        - --topics_path - A .jsonl file to read topics from. Defaults to 'topics.jsonl'
        - --output_path - Where to write the dataset when finished. Can be .jsonl or .csv. Defaults to a timestamped csv in the data/ folder.
        - --n_topics - How many topics from the topics file to use. Leave blank to use them all. For testing you should specify something in the range 1-3.
        - --n_subtopics - How many subtopics to generate per topic. 50 works well. Default is 5 for testing.
        - --n_prompts - How many prompts to generate per subtopic. 10 works well. Default is 5 for testing.
- Load a dataset from a file: `dataset = GeneratedPromptResponseDataset.from_file(path)` - returns a pytorch dataset.
- Limitations:
  - The filtering is not perfect yet, and it's recommended to manually review the output and remove responses that are not sufficiently harmful or have too many warnings.
  - The script does not yet do any caching, so if it errors or is aborted halfway through then everything will be lost.
  - It does not do calls concurrently yet and so is quite slow.
- Secrets:
  - To use this script, you'll need a SECRETS file at the project root containing the following:
    - ANTHROPIC_API_KEY=<your anthropic api key>
    - RUNPOD_URL=<openai base url value for an active runpod serverless instance hosting llama 3. We currently have one running here: https://www.runpod.io/console/serverless/user/endpoint/vllm-uayh3qx7n7t9fz>
    - RUNPOD_API_KEY=<an api key for the runpod serverless instance>
