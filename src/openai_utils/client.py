import base64
from pathlib import Path
import httpx
import requests
import os


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
    ) -> str | None:
        # Note: you can get back the logprobs too, but currently not parsing it out
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
            "logprobs": True,
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
            "logprobs": True,
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


async def main():
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
    client = OpenAIClient(api_key)
    response = await client.a_call_gpt_4_turbo(
        question, base64_image, temperature, max_tokens
    )
    print(response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
