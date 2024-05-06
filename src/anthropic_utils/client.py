from abc import ABC, abstractmethod
import base64
import hashlib
from pathlib import Path
from threading import Lock
from typing import Generic, Sequence, Type, TypeVar
from pydantic import BaseModel
import anthropic
from retry import retry


# Function to encode the image
def encode_image(image_path: Path | str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class InferenceConfig(BaseModel):
    # Config for openai
    model: str
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 100
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    n: int = 1


class ChatMessage(BaseModel):
    role: str
    content: str
    # base64
    image_content: str | None = None
    image_type: str | None = None # image/jpeg

    def to_anthropic_content(self) -> dict:
        if not self.image_content:
            return {
            "role": self.role,
            "content": [
                {"type": "text", "text": self.content},
            ],
            }
        else: 
            """
                            {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image1_media_type,
                        "data": image1_data,
                    },
                },
            """
            return {
            "role": self.role,
            "content": [
                {"type": "text", "text": self.content},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": self.image_type or "image/jpeg",
                        "data": self.image_content,
                    },
                }
            ],
        }


class InferenceResponse(BaseModel):
    raw_responses: Sequence[str]

    @property
    def single_response(self) -> str:
        if len(self.raw_responses) != 1:
            raise ValueError(
                f"This response has multiple responses {self.raw_responses}"
            )
        else:
            return self.raw_responses[0]




class FileCacheRow(BaseModel):
    key: str
    response: str # Should be generic, but w/e


def write_jsonl_file_from_basemodel(
    path: Path | str, basemodels: Sequence[BaseModel]
) -> None:
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for basemodel in basemodels:
            f.write(basemodel.model_dump_json() + "\n")


GenericBaseModel = TypeVar("GenericBaseModel", bound=BaseModel)


def read_jsonl_file_into_basemodel(
    path: Path | str, basemodel: Type[GenericBaseModel]
) -> list[GenericBaseModel]:
    with open(path) as f:
        return [basemodel.model_validate_json(line) for line in f]


class APIRequestCache:
    def __init__(self, cache_path: Path | str, response_type: Type[BaseModel]):
        self.cache_path = Path(cache_path)
        self.data: dict[str, BaseModel] = {}
        if self.cache_path.exists():
            rows = read_jsonl_file_into_basemodel(
                path=self.cache_path,
                basemodel=FileCacheRow,
            )
        else:
            rows = []
        print(f"Loaded {len(rows)} rows from cache file {self.cache_path.as_posix()}")
        self.response_type = response_type
        self.data: dict[str, response_type] = {row.key: response_type.model_validate_json(row.response) for row in rows}
        self.file_handler = self.cache_path.open("a")

    def write_line(self, key: str, response_json: str) -> None:
        self.file_handler.write(
            FileCacheRow(key=key, response=response_json).model_dump_json() + "\n"
        )

    def __getitem__(self, key: str) -> BaseModel:
        return self.data[key]

    def __setitem__(self, key: str, value: BaseModel) -> None:
        self.data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __delitem__(self, key: str) -> None:
        del self.data[key]


def deterministic_hash(something: str) -> str:
    return hashlib.sha1(something.encode()).hexdigest()


def file_cache_key(messages: Sequence[ChatMessage], config: InferenceConfig) -> str:
    str_messages = ",".join([str(msg) for msg in messages]) + deterministic_hash(
        config.model_dump_json()
    )
    return deterministic_hash(str_messages)

# Generic to say what we are caching
CachedResponse = TypeVar("CachedResponse", bound=BaseModel)

from anthropic.types import Message as AnthropicResponse

class ModelCaller(ABC, Generic[CachedResponse]):

    @abstractmethod
    def cached_call(
        self,
        messages: Sequence[ChatMessage],
        config: InferenceConfig,
        try_number: int = 1,
    ) -> CachedResponse:
        raise NotImplementedError()

    def with_file_cache(self, cache_path: Path | str, response_type: Type[CachedResponse]) -> "CachedCaller[CachedResponse]":
        """
        Load a file cache from a path
        Alternatively, rather than write_every_n, just dump with append mode?
        """
        if isinstance(cache_path, str):
            cache_path = Path(cache_path)

        return CachedCaller(wrapped_caller=self, cache_path=cache_path,response_type=response_type)



class CachedCaller(ModelCaller[CachedResponse]):
    def __init__(self, wrapped_caller: ModelCaller, cache_path: Path | str, response_type: Type[CachedResponse]):
        self.model_caller = wrapped_caller
        self.cache_path = cache_path
        self.cache: APIRequestCache = APIRequestCache(cache_path=cache_path, response_type=response_type)
        self.save_lock = Lock()
        self.response_type = response_type

    def cached_call(
        self,
        messages: Sequence[ChatMessage],
        config: InferenceConfig,
        try_number: int = 1,
    ) -> CachedResponse:
        key_without_retry = file_cache_key(messages, config)
        # only add retry number to key if try_number > 1 for backwards compatibility
        key = (
            key_without_retry
            if try_number == 1
            else f"{key_without_retry}_try_{try_number}"
        )
        if key in self.cache:
            return self.cache[key] # type: ignore
        else:
            # uncahced call
            response = self.model_caller.cached_call(messages, config)
            value = response
            self.cache[key] = value
            response_str = value.model_dump_json()
            with self.save_lock:
                self.cache.write_line(key=key, response_json=response_str)

            return response



class AnthropicCaller(ModelCaller[AnthropicResponse]):
    def __init__(self, client: anthropic.Anthropic | None = None):
        self.client = client if client is not None else anthropic.Anthropic()

    @property
    def response_type(self) -> Type[AnthropicResponse]:
        return AnthropicResponse

    @retry(
        exceptions=(anthropic.APIConnectionError, anthropic.InternalServerError),
        tries=10,
        delay=5,
    )
    @retry(exceptions=(anthropic.RateLimitError), tries=-1, delay=1)
    def cached_call(
        self,
        messages: Sequence[ChatMessage],
        config: InferenceConfig,
        try_number: int = 1,
    ) -> AnthropicResponse:
        assert (
            "claude" in config.model
        ), f"AnthropicCaller can only be used with claude models. Got {config.model}"
        
        resp: AnthropicResponse = self.client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,  # type: ignore
            messages=[
                msg.to_anthropic_content()
                for msg in messages # type: ignore
            ],
        )
        return resp


def main():
    import dotenv

    api_key = dotenv.dotenv_values()["ANTHROPIC_API_KEY"]
    assert api_key is not None, "Please set the ANTHROPIC_API_KEY in your .env file"
    # Create a client
    client = anthropic.Anthropic(api_key=api_key)
    caller = AnthropicCaller(client=client).with_file_cache(cache_path="cache.jsonl", response_type=AnthropicResponse)
    # Create a config
    # model="claude-3-opus-20240229" #
    # model="claude-3-sonnet-20240229"  #
    model = "claude-3-haiku-20240307"
    config = InferenceConfig(model=model)
    # Path to your image
    image_path = "images/trina/000.jpg"

    # Getting the base64 string
    base64_image = encode_image(image_path)
    question = "What is the animal?"
    # Create a message
    message = ChatMessage(role="user", content=question, image_content=base64_image)
    # Call the model
    response = caller.cached_call(messages=[message], config=config)
    print(response.content[0].text)


if __name__ == "__main__":
    main()
