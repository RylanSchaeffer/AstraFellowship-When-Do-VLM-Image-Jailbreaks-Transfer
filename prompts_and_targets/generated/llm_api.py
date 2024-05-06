import asyncio
import os
from itertools import chain
from pathlib import Path
from typing import Callable, Optional
from enum import Enum, auto

from termcolor import cprint
import attrs
from anthropic import AsyncAnthropic

import json
import time
from datetime import datetime
from traceback import format_exc

from anthropic.types.content_block import ContentBlock as AnthropicContentBlock


def load_secrets(file_path):
    secrets = {}
    with open(file_path) as f:
        for line in f:
            key, value = line.strip().split("=", 1)
            secrets[key] = value
    return secrets


PRINT_COLORS = {"user": "cyan", "system": "magenta", "assistant": "cyan"}

OAIChatPrompt = list[dict[str, str]]


class StopReason(Enum):
    MAX_TOKENS = auto()
    STOP_SEQUENCE = auto()

    @classmethod
    def factory(cls, stop_reason: str) -> "StopReason":
        """
        Parses the openai and anthropic stop reasons into a StopReason enum.
        """
        if stop_reason in ["max_tokens", "length"]:
            return cls.MAX_TOKENS
        elif stop_reason in ["stop_sequence", "stop", "end_turn"]:
            return cls.STOP_SEQUENCE
        raise ValueError(f"Invalid stop reason: {stop_reason}")

    def __repr__(self):
        return self.name


@attrs.frozen()
class LLMResponse:
    model_id: str
    completion: str
    stop_reason: StopReason = attrs.field(converter=StopReason.factory)
    duration: Optional[float] = None
    api_duration: Optional[float] = None
    logprobs: Optional[list[dict[str, float]]] = None

    def to_dict(self):
        return {
            "model_id": self.model_id,
            "completion": self.completion,
            "stop_reason": self.stop_reason.__repr__(),  # Convert to some JSON-serializable format.
            "duration": self.duration,
            "api_duration": self.api_duration,
            "logprobs": self.logprobs,
        }


ANTHROPIC_MODELS = {
    "claude-3-opus-20240229",
}


def extract_system_prompt(messages: list) -> str:
    sys_prompt = ""
    for message in messages:
        if message["role"] == "system":
            if sys_prompt:
                raise ValueError(
                    "Multiple system messages found in the prompt. Only one is allowed."
                )
            sys_prompt = message["content"]
    return sys_prompt


def transform_messages(messages: list) -> list:
    _messages = []
    for message in messages:
        role = message["role"]

        if role == "system":
            continue
        content = message["content"]
        _messages.append({"role": role, "content": [{"type": "text", "text": content}]})
    return _messages


class AnthropicChatModel:
    def __init__(self, num_threads, print_prompt_and_response):
        super().__init__()
        secrets = load_secrets("SECRETS")
        api_key = secrets["ANTHROPIC_API_KEY"]
        self.num_threads = num_threads
        self.print_prompt_and_response = print_prompt_and_response
        self.client = AsyncAnthropic(api_key=api_key)
        self.available_requests = asyncio.BoundedSemaphore(int(self.num_threads))

    @staticmethod
    def _create_prompt_history_file(prompt):
        filename = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}_prompt.txt"
        with open(os.path.join("prompt_history", filename), "w") as f:
            json_str = json.dumps(prompt, indent=4)
            json_str = json_str.replace("\\n", "\n")
            f.write(json_str)

        return filename

    @staticmethod
    def _add_response_to_prompt_file(prompt_file, response):
        with open(os.path.join("prompt_history", prompt_file), "a") as f:
            f.write("\n\n======RESPONSE======\n\n")
            json_str = json.dumps(response.to_dict(), indent=4)
            json_str = json_str.replace("\\n", "\n")
            f.write(json_str)

    async def __call__(
        self,
        model_id: str,
        prompt: OAIChatPrompt,
        max_attempts: int = 10,
        is_valid: Callable = lambda _: True,
        max_tokens: int = 2048,
        **kwargs,
    ) -> list[LLMResponse]:
        start = time.time()
        print(f"Queuing {model_id} call")
        response: Optional[AnthropicContentBlock] = None
        duration = None
        system_prompt = extract_system_prompt(prompt)
        prompt = transform_messages(prompt)
        # prompt_file = self._create_prompt_history_file([system_prompt] + prompt)
        for i in range(max_attempts):
            try:
                async with self.available_requests:
                    api_start = time.time()
                    print(f"Making {model_id} call")
                    response = await self.client.messages.create(
                        messages=prompt,
                        model=model_id,
                        system=system_prompt,
                        max_tokens=max_tokens,
                        **kwargs,
                    )
                    api_duration = time.time() - api_start
                    if not is_valid(response.content[0].text):
                        print(f"Invalid API response: {response}")
                        response = None
                        raise Exception
            except Exception as e:
                error_info = f"Exception Type: {type(e).__name__}, Error Details: {str(e)}, Traceback: {format_exc()}"
                print(
                    f"Encountered API error: {error_info}.\nRetrying now. (Attempt {i})"
                )
                await asyncio.sleep(1.5**i)
            else:
                break

        if response is None:
            raise RuntimeError(
                f"Failed to get a response from the API after {max_attempts} attempts."
            )

        duration = time.time() - start
        print(f"Completed call to {model_id} in {duration}s")

        llm_response = LLMResponse(
            model_id=model_id,
            completion=response.content[0].text,
            stop_reason=response.stop_reason,
            duration=duration,
            api_duration=api_duration,
        )

        # self._add_response_to_prompt_file(prompt_file, llm_response)
        if self.print_prompt_and_response:
            cprint("Prompt:", "white")
            cprint("System: " + system_prompt, PRINT_COLORS["system"])
            for message in prompt:
                role = message["role"]
                content = message["content"]
                tag = "Human: " if role == "user" else "Assistant: "
                cprint(tag, PRINT_COLORS[role], end="")
                cprint(content, PRINT_COLORS[role])

            cprint(f"Response ({llm_response.model_id}):", "white")
            cprint(
                f"{llm_response.completion}", PRINT_COLORS["assistant"], attrs=["bold"]
            )
            print()

        return [llm_response]


class ModelAPI:
    def __init__(
        self, num_threads: int = 5, print_prompt_and_response: bool = True
    ) -> None:
        self.num_threads = num_threads
        self.print_prompt_and_response = print_prompt_and_response
        self._anthropic_chat = AnthropicChatModel(
            num_threads=self.num_threads,
            print_prompt_and_response=self.print_prompt_and_response,
        )
        # Path("./prompt_history").mkdir(exist_ok=True)

    async def __call__(
        self,
        model_id: str,
        n: int = 1,
        **kwargs,
    ) -> list[LLMResponse]:
        if model_id in ANTHROPIC_MODELS:
            client = self._anthropic_chat
        else:
            raise ValueError(f"Invalid model id: {model_id}")

        responses = list(
            chain.from_iterable(
                await asyncio.gather(
                    *[
                        client(
                            model_id=model_id,
                            **kwargs,
                        )
                        for _ in range(n)
                    ]
                )
            )
        )

        return responses[:n]
