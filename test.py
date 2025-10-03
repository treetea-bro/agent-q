import instructor
import litellm
from dotenv import load_dotenv
from instructor import Mode
from pydantic import BaseModel

# .env 로드
load_dotenv()


class Input(BaseModel):
    question: str


class Output(BaseModel):
    answer: str


# Instructor 클라이언트 (litellm 기반)
client = instructor.from_litellm(
    litellm.completion,
    mode=Mode.JSON,
)

messages = [{"role": "user", "content": "안녕, 나는 외부에서 접속하고 있어."}]

response = client.chat.completions.create(
    model="ollama/qwen3:32b",  # ✅ 반드시 prefix 포함
    messages=messages,
    response_model=Output,
)

res
print(response)


def add(a: int, b: int) -> int:
    return a + b
