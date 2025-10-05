import os
from typing import Callable, List, Optional, Tuple, Type

import instructor
import litellm
from instructor import Mode
from langsmith import traceable
from pydantic import BaseModel

from agentq.utils.function_utils import get_function_schema

model = os.getenv("MODEL")
vlm_model = os.getenv("VLM_MODEL")


class BaseAgent:
    def __init__(
        self,
        name: str,
        system_prompt: str,
        input_format: Type[BaseModel],
        output_format: Type[BaseModel],
        tools: Optional[List[Tuple[Callable, str]]] = None,
        keep_message_history: bool = True,
    ):
        # Metadata
        self.agent_name = name

        # Messages
        self.system_prompt = system_prompt
        if self.system_prompt:
            self._initialize_messages()
        self.keep_message_history = keep_message_history

        self.input_format = input_format
        self.output_format = output_format

        self.client = instructor.from_litellm(
            litellm.completion,
            mode=Mode.JSON,
        )

        self.tools_list = []
        self.executable_functions_list = {}
        if tools:
            self._initialize_tools(tools)

    def _initialize_tools(self, tools: List[Tuple[Callable, str]]):
        for func, func_desc in tools:
            self.tools_list.append(get_function_schema(func, description=func_desc))
            self.executable_functions_list[func.__name__] = func

    def _initialize_messages(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]

    @traceable(run_type="chain", name="agent_run")
    async def run(
        self, input_data: BaseModel, screenshot: str = None, session_id: str = None
    ) -> BaseModel:
        if not isinstance(input_data, self.input_format):
            raise ValueError(f"Input data must be of type {self.input_format.__name__}")

        if not self.keep_message_history:
            self._initialize_messages()

        if screenshot:
            self.messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": input_data.model_dump_json(
                                exclude={"current_page_dom", "current_page_url"}
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": screenshot}},
                    ],
                }
            )
        else:
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
                    "content": f"Current page URL:\n{input_data.current_page_url}\n\n Current page DOM:\n{input_data.current_page_dom}",
                }
            )

        selectted_model = vlm_model if screenshot else model

        while True:
            if len(self.tools_list) == 0:
                print("111111111111111111111111111111111111111")
                response = self.client.chat.completions.create(
                    model=selectted_model,
                    messages=self.messages,
                    response_model=self.output_format,
                    max_retries=3,
                )
                print("222222222222222222222222222222222222222")
            else:
                response = self.client.chat.completions.create(
                    model=selectted_model,
                    messages=self.messages,
                    response_model=self.output_format,
                    tool_choice="auto",
                    tools=self.tools_list,
                )
            print("333333333333333333333333333333333333333")

            return response
