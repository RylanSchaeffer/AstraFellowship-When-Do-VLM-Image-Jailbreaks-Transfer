from PIL import Image
import requests
def load_pil_image_from_url(path: str):
    if path.startswith("http://") or path.startswith("https://"):
        image = Image.open(requests.get(path, stream=True).raw)
    else:
        image = Image.open(path)
    image = image.convert("RGB")
    return image
    