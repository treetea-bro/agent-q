import json
from typing import Callable, List, Optional, Tuple, Type

import torch
from pydantic import BaseModel, ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer

from agentq.utils.function_utils import get_function_schema


class BaseAgent:
    def __init__(
        self,
        name: str,
        system_prompt: str,
        input_format: Type[BaseModel],
        output_format: Type[BaseModel],
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        tools: Optional[List[Tuple[Callable, str]]] = None,
        keep_message_history: bool = True,
    ):
        self.agent_name = name
        self.system_prompt = system_prompt
        self.keep_message_history = keep_message_history
        self.input_format = input_format
        self.output_format = output_format
        self.model = model
        self.tokenizer = tokenizer

        if self.system_prompt:
            self._initialize_messages()

        self.tools_list = []
        self.executable_functions_list = {}
        if tools:
            self._initialize_tools(tools)

    def update_model(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        print(f"[BaseAgent] Model updated in-memory: {self.agent_name}")
        self.model = model
        self.tokenizer = tokenizer

    def _initialize_tools(self, tools: List[Tuple[Callable, str]]):
        for func, func_desc in tools:
            self.tools_list.append(get_function_schema(func, description=func_desc))
            self.executable_functions_list[func.__name__] = func

    def _initialize_messages(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]

    async def run(
        self,
        input_data: BaseModel,
        screenshot: str | None = None,
        session_id: str | None = None,
    ):
        if not isinstance(input_data, self.input_format):
            raise ValueError(f"Input data must be of type {self.input_format.__name__}")

        if not self.keep_message_history:
            self._initialize_messages()

        # === Prompt 구성 ===
        prompt = input_data.model_dump_json(
            exclude={"current_page_dom", "current_page_url"}
        )
        if hasattr(input_data, "current_page_dom") and hasattr(
            input_data, "current_page_url"
        ):
            prompt += f"\n\nCurrent page URL:\n{input_data.current_page_url}\n\nDOM:\n{input_data.current_page_dom}"

        # === JSON Schema 안내 ===
        schema_instruction = f"""
        You must answer strictly in JSON.
        The JSON must follow this Pydantic schema:

        {self.output_format.schema_json(indent=2)}
        """

        if self.system_prompt:
            full_prompt = (
                self.system_prompt + "\n\n" + schema_instruction + "\n\n" + prompt
            )
        else:
            full_prompt = schema_instruction + "\n\n" + prompt

        # === 모델 실행 ===
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # === JSON 파싱 시도 ===
        try:
            parsed = json.loads(decoded)
            return self.output_format.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"[BaseAgent] ⚠️ JSON 파싱 실패: {e}")
            return decoded
