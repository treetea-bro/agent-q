import json
import re
from typing import Callable, List, Optional, Tuple, Type

import torch
from langsmith import traceable
from pydantic import BaseModel

from agentq.utils.function_utils import get_function_schema
from agentq.utils.logger import logger


class BaseAgent:
    def __init__(
        self,
        model,
        tokenizer,
        name: str,
        system_prompt: str,
        input_format: Type[BaseModel],
        output_format: Type[BaseModel],
        tools: Optional[List[Tuple[Callable, str]]] = None,
        keep_message_history: bool = True,
    ):
        # Metadata
        self.model = model
        self.tokenizer = tokenizer
        self.agent_name = name

        # Messages
        self.system_prompt = system_prompt
        if self.system_prompt:
            self._initialize_messages()
        self.keep_message_history = keep_message_history

        # I/O schemas
        self.input_format = input_format
        self.output_format = output_format

        # Tools
        self.tools_list = []
        self.executable_functions_list = {}
        if tools:
            self._initialize_tools(tools)

    def update_model(self, model):
        print(f"[BaseAgent] Model updated in-memory: {self.agent_name}")
        self.model = model

    def _initialize_tools(self, tools: List[Tuple[Callable, str]]):
        for func, func_desc in tools:
            self.tools_list.append(get_function_schema(func, description=func_desc))
            self.executable_functions_list[func.__name__] = func

    def _initialize_messages(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def _extract_assistant_json(self, text: str):
        """
        간소화 버전:
        - 'assistant' 이후 나오는 첫 번째 JSON 블록만 추출.
        - 정규식으로 JSON 블록 감지.
        """
        match = re.search(r"assistant[:\s\n]*({.*})", text, re.DOTALL | re.IGNORECASE)
        if not match:
            raise ValueError(
                f"[BaseAgent] No JSON found after 'assistant':\n{text[:300]}"
            )
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"[BaseAgent] JSON parsing failed: {e}\nExtracted:\n{json_str[:300]}"
            )

    @traceable(run_type="chain", name="agent_run")
    async def run(
        self,
        input_data: BaseModel,
        screenshot: str | None = None,
        session_id: str | None = None,
    ) -> BaseModel:
        if not isinstance(input_data, self.input_format):
            raise ValueError(f"Input data must be of type {self.input_format.__name__}")

        # Reset messages if history not kept
        if not self.keep_message_history:
            self._initialize_messages()

        self.messages.append(
            {
                "role": "user",
                "content": input_data.model_dump_json(
                    exclude={"current_page_dom", "current_page_url"}
                ),
            }
        )

        # Add DOM context if present
        if hasattr(input_data, "current_page_dom") and hasattr(
            input_data, "current_page_url"
        ):
            self.messages.append(
                {
                    "role": "user",
                    "content": f"Current page URL:\n{input_data.current_page_url}\n\nCurrent page DOM:\n{input_data.current_page_dom}",
                }
            )

        # === Build chat prompt ===
        chat_prompt = self.tokenizer.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )

        # === Tokenize & Generate ===
        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[1]  # 기존 prompt 길이

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5000,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        # === Decode only newly generated tokens ===
        generated_tokens = outputs[0][input_length:]
        decoded = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()

        # === Parse and validate ===
        parsed = self._extract_assistant_json(decoded)
        return self.output_format.model_validate(parsed)

    async def _append_tool_response(self, tool_call):
        function_name = tool_call.function.name
        function_to_call = self.executable_functions_list[function_name]
        function_args = json.loads(tool_call.function.arguments)
        try:
            function_response = await function_to_call(**function_args)
            self.messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": str(function_response),
                }
            )
        except Exception as e:
            logger.error(f"Error occurred calling the tool {function_name}: {str(e)}")
            self.messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": "Tool error: please modify parameters and retry.",
                }
            )
