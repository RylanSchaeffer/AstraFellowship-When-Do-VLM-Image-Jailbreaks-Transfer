from dataclasses import dataclass


@dataclass
class PromptAndTarget:
    prompt: str
    target: str
