import json
from datetime import datetime
from typing import Callable, List, Optional, Tuple, Type

from icecream import ic
from outlines.inputs import Chat
from pydantic import BaseModel

from agentq.utils.function_utils import get_function_schema


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

        if hasattr(input_data, "current_page_dom") and hasattr(
            input_data, "current_page_url"
        ):
            self.messages.append(
                {
                    "role": "user",
                    "content": f"Current page URL:\n{input_data.current_page_url}\n\nCurrent page DOM:\n{input_data.current_page_dom}",
                }
            )

        inputs = Chat(self.messages)

        start_time = datetime.now()
        print(f"🚀 Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        outputs = self.model(
            inputs,
            self.output_format,
            max_new_tokens=4092,
        )

        end_time = datetime.now()
        print(f"🏁 End:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱ Duration: {(end_time - start_time).total_seconds():.2f} seconds")

        ic(outputs)
        parsed = json.loads(outputs)

        del inputs
        del outputs

        return self.output_format.model_validate(parsed)
