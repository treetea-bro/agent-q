from dotenv import load_dotenv

load_dotenv()
# from unsloth import FastLanguageModel  # isort: skip  # noqa: E402

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["OMP_NUM_THREADS"] = "4"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
#
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.benchmark = False
import asyncio
import json
import os
import shutil
import tempfile
from typing import List, Tuple

import numpy as np
import outlines
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from playwright.async_api import Page
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DPOTrainer
from trl.trainer.dpo_config import DPOConfig
from trl.trainer.dpo_trainer import DPOTrainer

from agentq.core.agent.agentq_actor import AgentQActor
from agentq.core.agent.agentq_critic import AgentQCritic
from agentq.core.agent.base import BaseAgent
from agentq.core.agent.vision_agent import VisionAgent
from agentq.core.mcts.core.base import Reasoner, SearchConfig, WorldModel
from agentq.core.mcts.core.mcts import MCTS, MCTSResult
from agentq.core.models.models import (
    ActionType,
    AgentQActorInput,
    AgentQActorOutput,
    AgentQCriticInput,
    AgentQCriticOutput,
    BrowserAction,
    BrowserState,
    DPOAction,
    DPOPair,
    DPOState,
    TaskWithActions,
    VisionInput,
    VisionOutput,
)
from agentq.core.skills.click_using_selector import click
from agentq.core.skills.enter_text_and_click import enter_text_and_click
from agentq.core.skills.enter_text_using_selector import EnterTextEntry, entertext
from agentq.core.skills.get_dom_with_content_type import get_dom_with_content_type
from agentq.core.skills.get_screenshot import get_screenshot
from agentq.core.skills.get_url import geturl
from agentq.core.skills.open_url import openurl
from agentq.core.web_driver.playwright import PlaywrightManager

# ANSI color codes
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"


# @traceable(run_type="chain", name="mcts")
class BrowserWorldModel(WorldModel[BrowserState, BrowserAction, str]):
    def __init__(self, objective: str, vision: BaseAgent) -> None:
        super().__init__()
        self.objective = objective
        self.vision = vision
        print(
            f"{BLUE}[DEBUG] BrowserWorldModel initialized with objective: {self.objective}{RESET}"
        )

    async def init_state(self) -> BrowserState:
        # go to home page
        print(f"{GREEN}[DEBUG] GOING TO INIT STATE HOMEPAGE{RESET}")
        playwright_manager = PlaywrightManager()
        await playwright_manager.go_to_homepage()

        # initialzie dom and url
        initial_dom = await self.get_current_dom()
        initial_url = await self.get_current_url()
        print(f"{GREEN}[DEBUG] Initial state created - URL: {initial_url}{RESET}")

        return BrowserState(
            dom=initial_dom,
            url=initial_url,
            objective=self.objective,
            completed_tasks=[],
        )

    async def step(
        self, state: BrowserState, browser_action: BrowserAction
    ) -> Tuple[BrowserState, dict]:
        print(f"{YELLOW}[DEBUG] Executing step with action: {browser_action}{RESET}")
        new_dom, new_url = await self.execute_browser_action(browser_action)
        current_task = browser_action.task_with_action
        new_completed_tasks = state.completed_tasks + [current_task]
        new_state = BrowserState(
            dom=new_dom,
            url=new_url,
            objective=state.objective,
            completed_tasks=new_completed_tasks,
        )
        print(f"{GREEN}[DEBUG] New state after step - URL: {new_url}{RESET}")
        return new_state, {}

    async def is_terminal(self, state: BrowserState) -> bool:
        terminal = await is_terminal(state, self.vision)
        print(f"{CYAN}[DEBUG] is_terminal: {terminal}{RESET}")
        return terminal

    async def execute_browser_action(
        self, browser_action: BrowserAction
    ) -> Tuple[str, str]:
        action = browser_action.task_with_action.actions_to_be_performed[0]
        print(f"{YELLOW}[DEBUG] Executing browser action: {action.type}{RESET}")

        if action.type == ActionType.GOTO_URL:
            print(f"{CYAN}[DEBUG] Trying to go to url{RESET}")
            await openurl(url=action.website, timeout=action.timeout or 1)
            print(f"{CYAN}[DEBUG] Went to url{RESET}")
        elif action.type == ActionType.TYPE:
            entry = EnterTextEntry(
                query_selector=f"[mmid='{action.mmid}']",
                text=action.content,
            )
            print("entry", "-" * 40)
            print(entry)
            print("-" * 40)
            await entertext(entry)
            # await wait_for_navigation()
            print(f"{CYAN}[DEBUG] Typed text into element{RESET}")
        elif action.type == ActionType.CLICK:
            await click(
                selector=f"[mmid='{action.mmid}']",
                wait_before_execution=action.wait_before_execution or 2,
            )
            print(f"{CYAN}[DEBUG] Clicked element{RESET}")
        elif action.type == ActionType.ENTER_TEXT_AND_CLICK:
            await enter_text_and_click(
                text_selector=f"[mmid='{action.text_element_mmid}']",
                text_to_enter=action.text_to_enter,
                click_selector=f"[mmid='{action.click_element_mmid}']",
                wait_before_click_execution=action.wait_before_click_execution or 2,
            )
            # await wait_for_navigation()
            print(f"{CYAN}[DEBUG] Entered text and clicked element{RESET}")

        try:
            new_dom = await self.get_current_dom()
        except Exception as e:
            print(f"{RED}[DEBUG] Error getting DOM after action: {e}{RESET}")
            new_dom = "Error: Unable to retrieve DOM"

        try:
            new_url = await self.get_current_url()
        except Exception as e:
            print(f"{RED}[DEBUG] Error getting URL after action: {e}{RESET}")
            new_url = "Error: Unable to retrieve URL"

        print(f"{GREEN}[DEBUG] After action execution - New URL: {new_url}{RESET}")
        return new_dom, new_url

    async def get_current_dom(self) -> str:
        await wait_for_navigation()
        dom = await get_dom_with_content_type(content_type="all_fields")
        print(f"{CYAN}[DEBUG] Got current DOM (length: {len(dom)}){RESET}")
        return str(dom)[:4000]

    async def get_current_url(self) -> str:
        # await wait_for_navigation()
        url = await geturl()
        print(f"{CYAN}[DEBUG] Got current URL: {url}{RESET}")
        return url


class BrowserMCTSSearchConfig(SearchConfig[BrowserState, BrowserAction, str]):
    def __init__(self, actor: BaseAgent, critic: BaseAgent, vision: BaseAgent) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.vision = vision
        print(f"{BLUE}[DEBUG] BrowserMCTSSearchConfig initialized{RESET}")

    async def get_actions(self, state: BrowserState) -> List[BrowserAction]:
        print(f"{YELLOW}[DEBUG] Getting actions for current state{RESET}")
        actor_input: AgentQActorInput = AgentQActorInput(
            objective=state.objective,
            completed_tasks=state.completed_tasks,
            current_page_dom=state.dom,
            current_page_url=state.url,
        )
        print("actor_input", "-" * 40)
        print(actor_input)
        print("-" * 40)
        actor_output: AgentQActorOutput = await self.actor.run(actor_input)

        proposed_tasks_with_actions: List[TaskWithActions] = actor_output.proposed_tasks
        print(
            f"{CYAN}[DEBUG] Number of proposed tasks: {len(proposed_tasks_with_actions)}{RESET}"
        )

        ranked_actions = await self._rank_actions(state, proposed_tasks_with_actions)
        print(f"{CYAN}[DEBUG] Number of sorted actions: {len(ranked_actions)}{RESET}")

        return ranked_actions

    async def reward(
        self, state: BrowserState, action: BrowserAction, **kwargs
    ) -> Tuple[float, dict]:
        terminal_state = await is_terminal(state=state, vision=self.vision)
        if terminal_state:
            print(f"{GREEN}[DEBUG] Terminal state reached, reward: 1.0{RESET}")
            return 1.0, {}
        else:
            print(f"{RED}[DEBUG] Non-terminal state, reward: -0.01{RESET}")
            return -0.01, {}

    def fast_reward(
        self, state: BrowserState, action: BrowserAction
    ) -> tuple[float, dict]:
        return action.rank, {}

    async def _rank_actions(
        self, state: BrowserState, tasks: List[TaskWithActions]
    ) -> List[BrowserAction]:
        ranked_actions = []
        remaining_tasks = tasks.copy()
        total_tasks = len(remaining_tasks)

        print(f"{GREEN}[INFO] Sorting task via Critic now...")
        for iteration in range(total_tasks):
            if not remaining_tasks:
                break

            critic_input = AgentQCriticInput(
                objective=state.objective,
                completed_tasks=state.completed_tasks,
                tasks_for_eval=remaining_tasks,
                current_page_url=state.url,
                current_page_dom=state.dom,
            )

            critic_output: AgentQCriticOutput = await self.critic.run(critic_input)
            top_task = critic_output.top_task

            if top_task and top_task.actions_to_be_performed:
                rank = 1.0 / (iteration + 1)  # Higher rank for earlier iterations
                ranked_actions.append(
                    BrowserAction(task_with_action=top_task, rank=rank)
                )

                # Remove the top task from remaining tasks
                remaining_tasks = [
                    task for task in remaining_tasks if task.id != top_task.id
                ]
            else:
                print(
                    f"{MAGENTA}[DEBUG] Warning: No valid top task found in iteration {iteration}. Skipping.{RESET}"
                )

        print(f"{CYAN}[DEBUG] Sorted actions.")
        return ranked_actions


async def is_terminal(state: BrowserState, vision: BaseAgent) -> bool:
    print(f"{YELLOW}[DEBUG] Checking if state is terminal{RESET}")
    screenshot = await get_screenshot()
    vision_input: VisionInput = VisionInput(objective=state.objective)
    vision_output: VisionOutput = await vision.run(
        vision_input, screenshot, model="gpt-4o-mini"
    )
    print(f"{YELLOW}[DEBUG] Output of vision LLM {vision_output.is_terminal}{RESET}")
    return vision_output.is_terminal


class BrowserMCTSWrapper(Reasoner[BrowserState, BrowserAction, str]):
    def __init__(
        self,
        objective: str,
        actor: BaseAgent,
        critic: BaseAgent,
        vision: BaseAgent,
        n_iterations: int = 1,
        depth_limit: int = 1,
        exploration_weight: float = 1.0,
    ):
        world_model = BrowserWorldModel(objective, vision)
        search_config = BrowserMCTSSearchConfig(actor, critic, vision)
        search_algo = MCTS(
            n_iters=n_iterations,
            w_exp=exploration_weight,
            cum_reward=sum,
            calc_q=np.mean,
            simulate_strategy="max",
            output_strategy="max_reward",
            depth_limit=depth_limit,
        )
        super().__init__(world_model, search_config, search_algo)
        self.dpo_pairs = []
        print(
            f"{BLUE}[DEBUG] BrowserMCTSWrapper initialized with objective: {objective}{RESET}"
        )

    async def __call__(self) -> MCTSResult:
        print(f"{YELLOW}[DEBUG] Starting MCTS search{RESET}")
        result = await super().__call__("")
        return result

    @staticmethod
    def generate_dpo_pairs(result: MCTSResult) -> List[DPOPair]:
        dpo_pairs = []

        if result.trace_of_nodes is None or len(result.trace_of_nodes) < 2:
            print(f"{RED}[DEBUG] No valid path found{RESET}")
            return []

        print(f"{BLUE}[DEBUG] Printing rewards before generating dpo pairs")
        for i, node in enumerate(result.trace_of_nodes):
            print(f"{BLUE} {node.state.url} - {node.Q}")

        for i in range(len(result.trace_of_nodes) - 1):
            current_node = result.trace_of_nodes[i]
            next_node = result.trace_of_nodes[i + 1]

            if current_node.children:
                winning_action = next_node.action
                for child in current_node.children:
                    if child.action != winning_action:
                        dpo_pair = DPOPair(
                            state=DPOState(
                                dom=current_node.state.dom[
                                    :1000
                                ],  # Truncate DOM to first 1000 characters
                                objective=current_node.state.objective,
                            ),
                            winning_action=DPOAction(
                                description=winning_action.task_with_action.description,
                                action=winning_action.task_with_action.actions_to_be_performed[
                                    0
                                ],
                            ),
                            losing_action=DPOAction(
                                description=child.action.task_with_action.description,
                                action=child.action.task_with_action.actions_to_be_performed[
                                    0
                                ],
                            ),
                        )
                        dpo_pairs.append(dpo_pair)

        return dpo_pairs

    @staticmethod
    def print_result(result: MCTSResult):
        if result.trace is None or len(result.trace) == 0:
            print(f"{RED}[DEBUG] No valid path found{RESET}")
            return

        states, actions = result.trace
        print(f"{GREEN}[DEBUG] Path found:{RESET}")
        for i, (state, action) in enumerate(zip(states, actions)):
            print(f"{CYAN}[DEBUG] Step {i}{RESET}")
            print(f"{CYAN}[DEBUG]  URL: {state.url}{RESET}")
            print(
                f"{CYAN}[DEBUG]  Action Type: {action.task_with_action.actions_to_be_performed[0].type}{RESET}"
            )
            print(
                f"{CYAN}[DEBUG]  Action Description: {action.task_with_action.description}{RESET}"
            )
            print(
                f"{CYAN}[DEBUG]  Action Detail: {action.task_with_action} - {action}{RESET}"
            )

        print(f"{GREEN}[DEBUG] Final URL: {states[-1].url}{RESET}")
        print(f"{GREEN}[DEBUG] Cumulative reward: {result.cum_reward}{RESET}")
        print(f"{GREEN}[DEBUG] Total steps: {len(actions)}{RESET}")

    @staticmethod
    def print_dpo_pairs(dpo_pairs: List[DPOPair]):
        print(f"\n{MAGENTA}═══════════════ Generated DPO Pairs ═══════════════{RESET}")
        for i, dpo_pair in enumerate(dpo_pairs, 1):
            print(f"\n{CYAN}╔══ Pair {i} ══╗{RESET}")
            print(f"{YELLOW}┌─ State ─┐{RESET}")
            trimmed_dom = (
                dpo_pair.state.dom[:100] + "..."
                if len(dpo_pair.state.dom) > 100
                else dpo_pair.state.dom
            )
            print(f"{YELLOW}│ DOM:{RESET} {trimmed_dom}")
            print(f"{GREEN}┌─ Winning Action ─┐{RESET}")
            print(f"{GREEN}│ Description:{RESET} {dpo_pair.winning_action.description}")
            print(f"{GREEN}│ Action Type:{RESET} {dpo_pair.winning_action.action.type}")
            print(f"{RED}┌─ Losing Action ─┐{RESET}")
            print(f"{RED}│ Description:{RESET} {dpo_pair.losing_action.description}")
            print(f"{RED}│ Action Type:{RESET} {dpo_pair.losing_action.action.type}")
            print(f"{CYAN}╚{'═' * (len('══ Pair X ══') - 2)}╝{RESET}")
        print(f"\n{MAGENTA}═══════════════ End of DPO Pairs ═══════════════{RESET}")

    @staticmethod
    async def write_dpo_pairs_to_file(dpo_pairs: List[DPOPair], filename: str):
        """
        Write the generated DPO pairs to a JSONL file in a format optimized for DPO training scripts.
        """
        with open(filename, "w") as f:
            for pair in dpo_pairs:
                dpo_entry = {
                    "prompt": f"Objective: {pair.state.objective}\nCurrent DOM: {pair.state.dom[:1000]}...",
                    "chosen": f"Action: {pair.winning_action.action.model_dump_json()}\nDescription: {pair.winning_action.description}",
                    "rejected": f"Action: {pair.losing_action.action.model_dump_json()}\nDescription: {pair.losing_action.description}",
                }
                json.dump(dpo_entry, f)
                f.write("\n")  # Add a newline for JSONL format

        print(f"{GREEN}[INFO] DPO pairs written to {filename} in JSONL format{RESET}")

    async def is_terminal(self, state: BrowserState) -> bool:
        print(f"{YELLOW}[DEBUG] Checking if state is terminal{RESET}")
        screenshot = await get_screenshot()
        vision_input: VisionInput = VisionInput(objective=state.objective)
        vision_output: VisionOutput = await self.vision.run(
            vision_input, screenshot, model="gpt-4o-mini"
        )
        print(
            f"{YELLOW}[DEBUG] Output of vision LLM {vision_output.is_terminal}{RESET}"
        )
        return vision_output.is_terminal


async def wait_for_navigation(max_retries=3):
    for attempt in range(max_retries):
        try:
            playwright_manager = PlaywrightManager()
            page = await playwright_manager.get_current_page()
            await page.wait_for_load_state("domcontentloaded", timeout=30000)
            print(
                f"{GREEN}[DEBUG] Navigation successful on attempt {attempt + 1}{RESET}"
            )
            return
        except Exception as e:
            print(
                f"{YELLOW}[DEBUG] Navigation error on attempt {attempt + 1}: {str(e)}{RESET}"
            )
    print(f"{RED}[DEBUG] Navigation failed after {max_retries} attempts{RESET}")


def pairs_to_dataset(pairs: List[DPOPair]) -> Dataset:
    """
    Converts a list of DPOPair objects into a Hugging Face Dataset
    for DPO training.

    Each DPOPair contains:
        - state (objective, dom)
        - winning_action (description, action)
        - losing_action (description, action)

    Output format per example:
        {
            "prompt": "<objective + DOM>",
            "chosen": "<winning action JSON>",
            "rejected": "<losing action JSON>"
        }
    """
    rows = []

    for pair in pairs:
        prompt = f"Objective: {pair.state.objective}\nCurrent DOM: {pair.state.dom}"

        chosen_action = {
            "description": pair.winning_action.description,
            "action": json.loads(pair.winning_action.action.model_dump_json()),
        }
        rejected_action = {
            "description": pair.losing_action.description,
            "action": json.loads(pair.losing_action.action.model_dump_json()),
        }

        chosen_str = "Action: " + json.dumps(chosen_action, ensure_ascii=False)
        rejected_str = "Action: " + json.dumps(rejected_action, ensure_ascii=False)

        rows.append(
            {
                "prompt": prompt,
                "chosen": chosen_str,
                "rejected": rejected_str,
            }
        )

    # === 4️⃣ Dataset 변환 ===
    dataset = Dataset.from_list(rows)
    print(f"✅ Created DPO dataset with {len(rows)} examples")
    return dataset


# ============ QLoRA Helper ============


def print_trainable_params(model):
    trainable, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    pct = 100 * trainable / total
    print(
        f"{CYAN}[LoRA] Trainable params: {trainable:,} / {total:,} ({pct:.2f} %){RESET}"
    )


def build_qlora_policy(model_name: str, gpu_num: int = 0):
    """
    Automatically detects model quantization type and builds QLoRA policy accordingly.
    Handles:
        - MxFP4 / GPTQ / AWQ pre-quantized models → skip BitsAndBytesConfig
        - Standard FP16/BF16 models → use BitsAndBytesConfig (NF4)
    """

    print(f"{YELLOW}[INFO] Checking model config for quantization type...{RESET}")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    quant_type = getattr(config, "quantization_config", None)
    quant_str = str(quant_type).lower() if quant_type else "none"

    if any(q in quant_str for q in ["mxfp4", "gptq", "awq", "aqlm"]):
        print(
            f"{GREEN}[INFO] Detected pre-quantized model ({quant_type.__class__.__name__ if quant_type else 'Unknown'})"
        )
        print(f"{GREEN}[ACTION] Skipping BitsAndBytesConfig — loading as-is.{RESET}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": gpu_num},
            # device_map="auto",
            trust_remote_code=True,
        )

    else:
        print(f"{CYAN}[INFO] No quantization detected or standard FP model.{RESET}")
        print(
            f"{CYAN}[ACTION] Applying 4-bit NF4 quantization via BitsAndBytes.{RESET}"
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map={"": gpu_num},
            # device_map="auto",
            trust_remote_code=True,
        )

    # Prepare for LoRA fine-tuning
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # LoRA 설정
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # 디버그용 파라미터 출력
    print_trainable_params(model)

    print(f"{GREEN}[SUCCESS] QLoRA policy model built successfully!{RESET}")
    return model


def build_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


async def train_loop(
    objectives: list[str],
    model_name: str,
    eval_mode: bool = False,
    num_iterations: int = 3,
    output_dir: str = "./dpo_final",
):
    print(f"{BLUE}Starting QLoRA-DPO Loop{RESET}")
    playwright_manager = PlaywrightManager()

    # === 토크나이저 공용 ===
    tokenizer = build_tokenizer(model_name)

    # === 모델 두 개 분리 로드 ===
    # 추론 전용: GPU 1
    model_infer = build_qlora_policy(model_name, gpu_num=1)
    model_outline = outlines.from_transformers(model_infer, tokenizer)
    # 학습 전용: GPU 0
    # model_train = build_qlora_policy(model_name, gpu_num=0)

    # === 브라우저 초기화 ===
    if not eval_mode:
        await playwright_manager.async_initialize()
    else:
        await playwright_manager.async_initialize(
            eval_mode=eval_mode, homepage="http://localhost:3000/abc"
        )
        page: Page = await playwright_manager.get_current_page()
        await page.set_extra_http_headers({"User-Agent": "AgentQ-Bot"})

    print(f"{GREEN}Browser started and ready{RESET}")
    print(f"{BLUE}[DEBUG] Starting main function{RESET}")

    # 추론 에이전트는 GPU1의 model_infer 사용
    actor = AgentQActor(model_outline, tokenizer)
    critic = AgentQCritic(model_outline, tokenizer)
    vision = VisionAgent()  # 외부 API 기반이면 GPU 미사용

    # ---- LoRA 어댑터 동기화 유틸 ----
    def sync_lora_adapters(src_peft_model, dst_peft_model):
        """
        src(학습모델)의 LoRA 어댑터 가중치를 임시 디렉토리에 저장 후
        dst(추론모델)에 로드한다. PEFT 호환 방식을 사용해 안전하게 동기화.
        """
        tmpdir = tempfile.mkdtemp(prefix="lora_sync_")
        try:
            # LoRA 어댑터만 저장 (merge 하지 않음)
            src_peft_model.save_pretrained(tmpdir)
            # 추론모델에 어댑터 로드
            # 최신 peft는 load_adapter 지원. 없으면 state_dict로 대체.
            try:
                dst_peft_model.load_adapter(
                    tmpdir, adapter_name="default", is_trainable=False
                )
                if hasattr(dst_peft_model, "set_adapter"):
                    dst_peft_model.set_adapter("default")
            except Exception:
                # fallback: 상태 dict 직접 로드 (strict=False)
                dst_peft_model.load_state_dict(
                    src_peft_model.state_dict(), strict=False
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    last_trainer = None
    for i, objective in enumerate(objectives):
        browser_mcts_wrapper = BrowserMCTSWrapper(
            objective=objective,
            actor=actor,
            critic=critic,
            vision=vision,
            n_iterations=10,
            depth_limit=6,
            exploration_weight=1.0,
        )

        print(f"\n========== LOOP {i + 1}/{num_iterations} ==========")

        # ---- 1) MCTS로 데이터 수집 (GPU1 / model_infer) ----
        result = await browser_mcts_wrapper()
        BrowserMCTSWrapper.print_result(result)

        dpo_pairs = BrowserMCTSWrapper.generate_dpo_pairs(result=result)
        BrowserMCTSWrapper.print_dpo_pairs(dpo_pairs=dpo_pairs)
        train_dataset = pairs_to_dataset(dpo_pairs)

        dpo_args = DPOConfig(
            output_dir=None,
            save_strategy="no",
            logging_strategy="steps",
            logging_steps=10,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            bf16=True,
            beta=0.3,
            optim="paged_adamw_8bit",
            lr_scheduler_type="cosine",
            learning_rate=2e-5,
            warmup_ratio=0.05,
            remove_unused_columns=False,
            report_to=["none"],
        )

        trainer = DPOTrainer(
            model=model_infer,
            ref_model=None,
            args=dpo_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )

        trainer.train()
        last_trainer = trainer

        print(f"{YELLOW}[INFO] Syncing LoRA adapters: train(GPU0) → infer(GPU1){RESET}")
        # sync_lora_adapters(trainer.model, model_infer)

        actor.update_model(model_infer)

        del trainer

    if last_trainer is not None:
        os.makedirs(output_dir, exist_ok=True)
        last_trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"✅ 최종 LoRA 어댑터 포함 모델 저장 → {output_dir}")


# Temp class to write output to a file
class StreamToFile:
    def __init__(self, filename):
        self.file = open(filename, "w", buffering=1)

    def write(self, data):
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()


def build_unsloth_policy(model_name: str, max_seq_len: int = 4096):
    print(f"{YELLOW}[INFO] Loading {model_name} via Unsloth (QLoRA 4-bit){RESET}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_len,
        load_in_4bit=True,  # NF4 quantization
        dtype=None,  # Auto: bf16 / fp16
        # device_map="auto",
        device_map={"": 1},
    )

    # LoRA 어댑터 설정
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"{GREEN}[SUCCESS] Unsloth policy built successfully!{RESET}")
    return model, tokenizer


# =====================
# Main Train Loop
# =====================
async def train_loop_unsloth(
    objectives: list[str],
    model_name: str,
    output_dir: str = "./dpo_final_unsloth",
    eval_mode: bool = False,
):
    print(f"{BLUE}Starting Unsloth QLoRA-DPO Loop{RESET}")
    playwright_manager = PlaywrightManager()
    await playwright_manager.async_initialize()

    # ---- 모델 / 토크나이저 로드 ----
    model_train, tokenizer = build_unsloth_policy(model_name)

    # ---- 에이전트 초기화 ----
    actor = AgentQActor(model_train, tokenizer)
    critic = AgentQCritic(model_train, tokenizer)
    vision = VisionAgent()

    def sync_lora_adapters(src_peft_model, dst_peft_model):
        """LoRA adapter weight sync"""
        import shutil
        import tempfile

        tmpdir = tempfile.mkdtemp(prefix="lora_sync_")
        try:
            src_peft_model.save_pretrained(tmpdir)
            try:
                dst_peft_model.load_adapter(
                    tmpdir, adapter_name="default", is_trainable=False
                )
                if hasattr(dst_peft_model, "set_adapter"):
                    dst_peft_model.set_adapter("default")
            except Exception:
                dst_peft_model.load_state_dict(
                    src_peft_model.state_dict(), strict=False
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    last_trainer = None
    for i, objective in enumerate(objectives):
        print(f"\n========== LOOP {i + 1}/{len(objectives)} ==========")
        browser_mcts_wrapper = BrowserMCTSWrapper(
            objective=objective,
            actor=actor,
            critic=critic,
            vision=vision,
            n_iterations=10,
            depth_limit=6,
            exploration_weight=1.0,
        )

        # === 1️⃣ MCTS로 DPO Pair 생성 ===
        result = await browser_mcts_wrapper()
        dpo_pairs = BrowserMCTSWrapper.generate_dpo_pairs(result)
        train_dataset = pairs_to_dataset(dpo_pairs)

        # === 2️⃣ DPO 학습 ===
        print(f"{YELLOW}[INFO] Training DPO with Unsloth Trainer{RESET}")
        training_args = TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            num_train_epochs=1,
            bf16=True,
            optim="paged_adamw_32bit",
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            remove_unused_columns=False,
            report_to=["none"],
            logging_steps=10,
            output_dir=output_dir,
            save_strategy="no",
        )

        trainer = DPOTrainer(
            model=model_train,
            ref_model=None,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            args=training_args,
        )

        trainer.train()
        last_trainer = trainer

        print(f"{YELLOW}[INFO] Syncing LoRA adapters (train → infer){RESET}")
        sync_lora_adapters(trainer.model, model_train)
        actor.update_model(model_train)

        del trainer

    if last_trainer is not None:
        os.makedirs(output_dir, exist_ok=True)
        last_trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"✅ 최종 LoRA 어댑터 포함 모델 저장 → {output_dir}")


if __name__ == "__main__":
    objectives = [
        "Play the latest episode of Friends.",
        "Play the most viewed Pokemon movie.",
        "Play the most viewed movie on YouTube.",
    ]
    provider = "hf"
    print(f"{BLUE}[DEBUG] Script started{RESET}")
    if provider == "hf":
        asyncio.run(
            train_loop(
                objectives=objectives,
                # model_name="openai/gpt-oss-20b",
                # model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
                model_name="Qwen/Qwen3-4B-Instruct-2507",
                # model_name="Qwen/Qwen2.5-14B-Instruct",
            )
        )
    elif provider == "unsloth":
        asyncio.run(
            train_loop_unsloth(
                objectives,
                "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
                # "unsloth/Qwen3-14B-unsloth-bnb-4bit",
                # "unsloth/Qwen3-32B-unsloth-bnb-4bit",
                # model_name="Qwen/Qwen3-30B-A3B-Instruct",
            )
        )
    print(f"{GREEN}[DEBUG] Script finished{RESET}")
