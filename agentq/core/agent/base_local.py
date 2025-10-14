import json
import re
from datetime import datetime
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
        Robust JSON extractor:
        - Cleans up system tokens (<think>, <|...|>, assistantfinal)
        - Finds last '{' and parses till matching '}'
        - If parsing fails, attempts truncated repair
        """
        cleaned = text.strip()

        # 1️⃣ Remove noisy tokens
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"<\|.*?\|>", "", cleaned)
        cleaned = cleaned.replace("assistantfinal", "")
        cleaned = cleaned.replace("<|return|>", "")
        cleaned = cleaned.strip()

        # 2️⃣ Try to extract the last JSON object heuristically
        last_brace = cleaned.rfind("{")
        if last_brace == -1:
            logger.error("❌ No '{' found in model output")
            return None

        candidate = cleaned[last_brace:]

        # 3️⃣ First attempt — direct parse
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ Direct JSON parse failed: {e}. Attempting repair...")

        # 4️⃣ Repair mode — truncate to last valid closing brace
        closing_idx = candidate.rfind("}")
        if closing_idx != -1:
            try:
                fixed = candidate[: closing_idx + 1]
                return json.loads(fixed)
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON parse failed after repair: {e}\n{fixed[-300:]}")
                return None

        logger.error("❌ No closing brace found for JSON")
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

        inputs = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            reasoning_effort="low",
        ).to(self.model.device)

        start_time = datetime.now()
        print(f"🚀 Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        outputs = self.model.fast_generate(
            **inputs,
            max_new_tokens=2048,
        )
        end_time = datetime.now()
        print(f"🏁 End:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱ Duration: {(end_time - start_time).total_seconds():.2f} seconds")

        generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)

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
