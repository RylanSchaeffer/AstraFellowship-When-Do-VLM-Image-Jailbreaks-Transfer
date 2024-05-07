import base64
from pathlib import Path
from typing import Sequence
import httpx
from openai import BaseModel
import requests
import os
from pydantic import ValidationError
from retry import retry

from src.anthropic_utils.client import APIRequestCache, ChatMessage, InferenceConfig


# Function to encode the image
def encode_image(image_path: Path | str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class OpenAIClient:
    # We need to DYI this because currently the openai python api doesn't support you directly uploading an image lol
    def __init__(self, api_key: str):
        assert api_key, "Please provide an OpenAI API Key"
        self.api_key = api_key
        # reuse the session
        self.session = httpx.AsyncClient()

    def call_gpt_4_turbo(
        self,
        question: str,
        image_base_64: str,
        temperature: float = 0.0,
        max_tokens: int = 1,
        image_type: str = "image/jpeg",  # or image/png
    ) -> str | None:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": "gpt-4-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{question}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image_type};base64,{image_base_64}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        # use "https://api.openai.com/v1/chat/completions"
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=None,
        )

        try:
            json = response.json()
            return json["choices"][0]["message"]["content"]
        except Exception as e:
            print(e)
            return None

    # async version
    async def a_call_gpt_4_turbo(
        self,
        question: str,
        image_base_64: str,
        temperature: float = 0.0,
        max_tokens: int = 1,
    ) -> str | None:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": "gpt-4-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{question}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base_64}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        # use "https://api.openai.com/v1/chat/completions"
        response = await self.session.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=None,
        )
        json = response.json()
        return json["choices"][0]["message"]["content"]


class OpenaiResponse(BaseModel):
    choices: list[dict]
    usage: dict
    created: int
    model: str
    id: str
    system_fingerprint: str


class OpenAICachedCaller:
    def __init__(self, api_key: str, cache_path: Path | str):
        self.api_key = api_key
        self.cache: APIRequestCache[OpenaiResponse] = APIRequestCache(
            cache_path=cache_path, response_type=OpenaiResponse
        )

    def call_gpt_4_turbo(
        self,
        question: str,
        image_base_64: str,
        temperature: float = 0.0,
        max_tokens: int = 1,
        image_type: str = "image/jpeg",  # or image/png
    ) -> str | None:
        # Simple no BS way of calling with an image
        message = [
            ChatMessage(
                role="user",
                content=question,
                image_content=image_base_64,
                image_type=image_type,
            )
        ]
        result = self.call(
            messages=message,
            config=InferenceConfig(
                temperature=temperature, max_tokens=max_tokens, model="gpt-4-turbo"
            ),
        )
        return result.choices[0]["message"]["content"]

    @retry(
        exceptions=(ValidationError),
        tries=5,
        delay=5,
    )
    # @retry(exceptions=(anthropic.RateLimitError), tries=-1, delay=1)
    def call(
        self,
        messages: Sequence[ChatMessage],
        config: InferenceConfig,
        try_number: int = 1,
    ) -> OpenaiResponse:

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": "gpt-4-turbo",
            "messages": [m.to_openai_content() for m in messages],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "logprobs": True,
        }
        # use "https://api.openai.com/v1/chat/completions"
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=None,
        )
        resp = OpenaiResponse.model_validate(response.json())

        if self.cache is not None:
            self.cache.add_model_call(messages, config, try_number, resp)
        return resp


def main():
    # pip install python-dotenv
    import dotenv

    # Please set your .env file with the OPENAI_API_KEY
    dotenv.load_dotenv()
    # OpenAI API Key
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "Please provide an OpenAI API Key"

    # Path to your image
    image_path = "images/trina/000.jpg"

    # Getting the base64 string
    base64_image = encode_image(image_path)
    question = "What is the animal?"
    max_tokens = 100
    temperature = 0.0
    cached_caller = OpenAICachedCaller(api_key=api_key, cache_path="cached.jsonl")

    # client = OpenAIClient(api_key)
    response = cached_caller.call_gpt_4_turbo(
        question, base64_image, temperature, max_tokens
    )
    print(response)


if __name__ == "__main__":
    main()
