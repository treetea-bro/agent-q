import io

import ollama
from PIL import Image

model = "gemma3:27b"

image = Image.open("objects.jpg")
image = image.resize((896, 896))

img_byte_arr = io.BytesIO()
image.save(img_byte_arr, format="JPEG")
img_bytes = img_byte_arr.getvalue()

stream = ollama.generate(
    model=model,
    prompt="이 이미지의 주요 객체를 나열해.",
    images=[img_bytes],
    stream=True,
)

for chunk in stream:
    print(chunk["response"], end="", flush=True)
