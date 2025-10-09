import json

import torch
from dotenv import load_dotenv
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()


class AgentQActorOutput(BaseModel):
    thought: str
    is_complete: bool
    final_response: str | None


def run(
    prompt: str,
    output_format: BaseModel,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5000,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # print("hihi : ", decoded)

    parsed = json.loads(decoded)
    # print("parsed : ", parsed)
    return output_format.model_validate(parsed)


MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
policy_model = AutoModelForCausalLM.from_pretrained(
    MODEL, device_map="auto", dtype="auto"
)
run("hi", AgentQActorOutput, policy_model, tokenizer)
