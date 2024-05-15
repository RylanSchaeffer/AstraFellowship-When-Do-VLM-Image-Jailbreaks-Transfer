from PIL import Image
import requests


def load_pil_image_from_url(path: str):
    if path.startswith("http://") or path.startswith("https://"):
        image = Image.open(requests.get(path, stream=True).raw)
    else:
        image = Image.open(path)
    image = image.convert("RGB")
    return image


def only_assistant_response(initial_prompt: str, response: str) -> str:
    starting_text = initial_prompt + "\nassistant\n"
    assert starting_text in response, f"Expected {starting_text} to be in {response}"
    # remove everything before and including the assistant token
    new_response = response.split(starting_text)[1]
    # # remove the final \n
    # new_response = new_response[:-1]
    return new_response
