import json
import re
from typing import Callable, List, Optional, Tuple, Type

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

    def _extract_json_from_output(self, text: str) -> Optional[dict]:
    """
    Removes reasoning (<think>...</think>) if present,
    then extracts the final JSON block from text.
    """
    # 1️⃣ Remove reasoning sections (if exist)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 2️⃣ Capture the *last JSON-like* block even if prefixed by other tokens (e.g. "assistantfinal")
    json_match = re.search(r"\{[\s\S]*\}\s*$", cleaned)
    if not json_match:
        # Try more permissive pattern in case assistantfinal is present
        json_match = re.search(r"assistantfinal\s*(\{[\s\S]*\})", cleaned)
        if not json_match:
            logger.error("No JSON found in model output")
            return None

    # 3️⃣ Extract group 1 if matched with prefix
    group = json_match.group(1) if json_match.lastindex else json_match.group(0)

    try:
        return json.loads(group)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}\nText:\n{group}")
        return None


    # @traceable(run_type="chain", name="agent_run")
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

        print("inputs!")
        inputs = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            reasoning_effort="low",
        ).to(self.model.device)
        print("inputs!")

        print("outputs!")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
        )
        print("outputs!")

        print("gento!")
        generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        print("gento!")
        decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        print("decoded", "-" * 50)
        print(decoded)
        print("-" * 50)

        # del decoded
        # del generated_tokens
        # del outputs
        # del inputs

        parsed = self._extract_json_from_output(decoded)
        print("parsed", "-" * 50)
        print(parsed)
        print("-" * 50)

        # === Parse and validate ===
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
