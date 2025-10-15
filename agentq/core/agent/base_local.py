import json
import re
from datetime import datetime
from typing import Callable, List, Optional, Tuple, Type

from jsonformer import Jsonformer
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
        1) 마지막 <|channel|>final<|message|> 블록만 스코프
        2) 그 블록 안에서 첫 '{'부터 '문자열/이스케이프를 고려'한 중괄호 매칭으로 끝 '}' 위치 찾기
        3) 해당 구간만 json.loads
        """
        if not text:
            logger.error("❌ Empty text.")
            return None

        raw = text

        # 1) 최종 final 블록만 스코프: 마지막 <|channel|>final<|message|> 기준으로 자르기
        FINAL_TAG = "<|channel|>final<|message|>"
        idx = raw.rfind(FINAL_TAG)
        if idx != -1:
            scoped = raw[idx + len(FINAL_TAG) :]
        else:
            # fallback: 그래도 못 찾으면 전체에서 시도 (하지만 권장하지 않음)
            scoped = raw

        # 2) <|return|> 있으면 거기까지 자르기 (뒤 꼬리 정리)
        end_tag = "<|return|>"
        end_idx = scoped.find(end_tag)
        if end_idx != -1:
            scoped = scoped[:end_idx]

        # 3) 채널 토큰 제거: 토큰만 제거 (내용은 남김)
        #    *여기에서 .*? 는 토큰 단위만 제거, DOTALL 금지로 라인 넘어 매치 방지
        scoped = re.sub(r"<\|[^|]*?\|>", "", scoped)
        scoped = scoped.replace("assistantfinal", "")
        scoped = re.sub(r"<think>.*?</think>", "", scoped, flags=re.DOTALL)
        scoped = scoped.strip()

        # 4) 스코프 내에서 첫 '{' 위치
        start = scoped.find("{")
        if start == -1:
            logger.error("❌ No '{' found in final block.")
            return None

        # 5) 문자열/이스케이프 인지하는 중괄호 매칭으로 종료 '}' 찾기
        i = start
        depth = 0
        in_str = False
        esc = False
        end = -1
        while i < len(scoped):
            ch = scoped[i]

            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            i += 1

        if end == -1:
            logger.error(
                "❌ Could not find matching closing '}' for JSON in final block."
            )
            # 그래도 마지막 '}' 까지 잘라서 한 번 시도
            last_brace = scoped.rfind("}")
            if last_brace != -1:
                candidate = scoped[start : last_brace + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as e:
                    logger.error(
                        f"❌ Fallback parse failed: {e}\n--- tail ---\n{candidate[-400:]}"
                    )
                    return None
            return None

        candidate = scoped[start : end + 1]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            logger.error(
                f"❌ JSON parse error: {e}\n--- scoped head ---\n{scoped[:200]}\n--- candidate tail ---\n{candidate[-400:]}"
            )
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

        json_schema = self.output_format.model_json_shema()
        Jsonformer(self.model, self.tokenizer, json_schema, inputs)
        outputs = Jsonformer()
        print("outputs", "-" * 50)
        print(outputs)
        print("-" * 50)
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
