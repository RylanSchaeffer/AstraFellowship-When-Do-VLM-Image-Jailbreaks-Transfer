from abc import ABC, abstractmethod
import base64
import hashlib
from pathlib import Path
from threading import Lock
from typing import Sequence, Type, TypeVar
from pydantic import BaseModel
import anthropic
from retry import retry

# Function to encode the image
def encode_image(image_path: Path | str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class OpenaiInferenceConfig(BaseModel):
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

class InferenceResponse(BaseModel):
    raw_responses: Sequence[str]
    @property
    def single_response(self) -> str:
        if len(self.raw_responses) != 1:
            raise ValueError(f"This response has multiple responses {self.raw_responses}")
        else:
            return self.raw_responses[0]


class CachedValue(BaseModel):
    response: InferenceResponse
    messages: Sequence[ChatMessage]
    config: OpenaiInferenceConfig



class FileCacheRow(BaseModel):
    key: str
    response: CachedValue

        
def write_jsonl_file_from_basemodel(path: Path | str, basemodels: Sequence[BaseModel]) -> None:
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for basemodel in basemodels:
            f.write(basemodel.model_dump_json() + "\n")


GenericBaseModel = TypeVar("GenericBaseModel", bound=BaseModel)

def read_jsonl_file_into_basemodel(path: Path | str, basemodel: Type[GenericBaseModel]) -> list[GenericBaseModel]:
    with open(path) as f:
        return [basemodel.model_validate_json(line) for line in f]

class APIRequestCache:
    def __init__(self, cache_path: Path | str):
        self.cache_path = Path(cache_path)
        self.data: dict[str, CachedValue] = {}
        self.load()

    def load(self, silent: bool = False) -> "APIRequestCache":
        """
        Load a file cache from a path
        """

        if self.cache_path.exists():
            rows = read_jsonl_file_into_basemodel(
                path=self.cache_path,
                basemodel=FileCacheRow,
            )
            if not silent:
                print(f"Loaded {len(rows)} rows from cache file {self.cache_path.as_posix()}")
            self.data = {row.key: row.response for row in rows}
        return self

    def save(self) -> None:
        """
        Save a file cache to a path
        """
        rows = [FileCacheRow(key=key, response=response) for key, response in self.data.items()]
        write_jsonl_file_from_basemodel(self.cache_path, rows)

    def __getitem__(self, key: str) -> CachedValue:
        return self.data[key]

    def __setitem__(self, key: str, value: CachedValue) -> None:
        self.data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __delitem__(self, key: str) -> None:
        del self.data[key]

def deterministic_hash(something: str) -> str:
    return hashlib.sha1(something.encode()).hexdigest()



def file_cache_key(messages: Sequence[ChatMessage], config: OpenaiInferenceConfig) -> str:
    str_messages = ",".join([str(msg) for msg in messages]) + deterministic_hash(config.model_dump_json())
    return deterministic_hash(str_messages)




class ModelCaller(ABC):
    @abstractmethod
    def call(
        self,
        messages: Sequence[ChatMessage],
        config: OpenaiInferenceConfig,
        try_number: int = 1,
    ) -> InferenceResponse:
        raise NotImplementedError()

    def with_file_cache(self, cache_path: Path | str, write_every_n: int = 20) -> "CachedCaller":
        """
        Load a file cache from a path
        Alternatively, rather than write_every_n, just dump with append mode?
        """
        if isinstance(cache_path, str):
            cache_path = Path(cache_path)

        return CachedCaller(wrapped_caller=self, cache_path=cache_path, write_every_n=write_every_n)
    
class CachedCaller(ModelCaller):
    def __init__(self, wrapped_caller: ModelCaller, cache_path: Path, write_every_n: int, silent_loading: bool = False):
        self.model_caller = wrapped_caller
        self.cache_path = cache_path
        self.cache: APIRequestCache = APIRequestCache(cache_path)
        self.write_every_n = write_every_n
        self.__update_counter = 0
        self.save_lock = Lock()

    def save_cache(self) -> None:
        self.cache.save()
    def call(
        self,
        messages: Sequence[ChatMessage],
        config: OpenaiInferenceConfig,
        try_number: int = 1,
    ) -> InferenceResponse:
        key_without_retry = file_cache_key(messages, config)
        # only add retry number to key if try_number > 1 for backwards compatibility
        key = key_without_retry if try_number == 1 else f"{key_without_retry}_try_{try_number}"
        if key in self.cache:
            return self.cache[key].response
        else:
            response = self.model_caller.call(messages, config)
            value = CachedValue(
                response=response,
                messages=messages,
                config=config,
            )
            with self.save_lock:
                self.cache[key] = value
                self.__update_counter += 1
                if self.__update_counter % self.write_every_n == 0:
                    self.save_cache()
            return response

    def __del__(self):
        self.save_cache()



class AnthropicCaller(ModelCaller):
    def __init__(self, client: anthropic.Anthropic | None = None):
        self.client = client if client is not None else anthropic.Anthropic()

    @retry(
        exceptions=(anthropic.APIConnectionError, anthropic.InternalServerError),
        tries=10,
        delay=5,
    )
    @retry(exceptions=(anthropic.RateLimitError), tries=-1, delay=1)
    def call(
        self,
        messages: Sequence[ChatMessage],
        config: OpenaiInferenceConfig,
        try_number: int = 1,
    ) -> InferenceResponse:
        assert "claude" in config.model, f"AnthropicCaller can only be used with claude models. Got {config.model}"
        message = ""
        for msg in messages:
            match msg.role:
                case "user":
                    message += f"{anthropic.HUMAN_PROMPT} {msg.content}"
                case "assistant":
                    message += f"{anthropic.AI_PROMPT} {msg.content}"
                case _:
                    raise ValueError(f"Unknown role {msg.role}")

        resp = self.client.completions.create(
            prompt=message,
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model=config.model,
            max_tokens_to_sample=config.max_tokens,
            temperature=config.temperature,  # type: ignore
        )
        inf_response = InferenceResponse(raw_responses=[resp.completion])
        return inf_response


def main():
    import dotenv
    api_key = dotenv.dotenv_values()["ANTHROPIC_API_KEY"]
    assert api_key is not None, "Please set the ANTHROPIC_API_KEY in your .env file"
    # Create a client
    client = anthropic.Anthropic()
    # Create a config
    config = OpenaiInferenceConfig(model="claude-3.0")
    # Create a message
    message = ChatMessage(role="user", content="What is the meaning of life?")
    # Call the model
    response = client.call(messages=[message], config=config)
    print(response.single_response)

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())





