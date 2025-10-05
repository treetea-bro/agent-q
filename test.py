import litellm
from dotenv import load_dotenv

# .env 로드
load_dotenv()

messages = [{"role": "user", "content": "안녕"}]

response = litellm.completion(
    model="ollama/qwen3:4b",  # ollama 모델
    messages=messages,
)

print(response["choices"][0]["message"]["content"])
