from dotenv import load_dotenv

load_dotenv()

import asyncio

# from agentq.utils.stream_to_file import stream_to_file
import os
import textwrap
from typing import List, Optional, Tuple

import numpy as np
from colorama import Fore, init
from datasets import Dataset
from pydantic import BaseModel
from pydantic.fields import Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.dpo_config import DPOConfig
from trl.trainer.dpo_trainer import DPOTrainer

from agentq.core.agent.base import BaseAgent
from agentq.core.agent.base_vision import BaseVisionAgent
from agentq.core.mcts.mcts import MCTS, MCTSResult
from agentq.core.models.models import (
    Action,
    ActionType,
    AgentQActorInput,
    AgentQActorOutput,
    AgentQCriticInput,
    AgentQCriticOutput,
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

init(autoreset=True)

objective = os.getenv("OBJECTIVE", "")
model = os.getenv("MODEL", "")
vlm_model = os.getenv("VLM_MODEL", "")


if not objective:
    raise ValueError("OBJECTIVE environment variable is not set.")


class BrowserState(BaseModel):
    dom: str
    url: str
    objective: str
    completed_tasks: Optional[List[TaskWithActions]]


class BrowserAction(BaseModel):
    action: Action
    rank: float = Field(description="The rank of this action, higher is better")


class BrowserWorldModel:
    def __init__(self, objective: str, vision: BaseAgent) -> None:
        super().__init__()
        self.objective = objective
        self.vision = vision

    async def init_state(self) -> BrowserState:
        initial_dom = await self.get_current_dom()
        initial_url = await self.get_current_url()
        print(f"{Fore.GREEN}[DEBUG] Initial state created - URL: {initial_url}")
        return BrowserState(
            dom=initial_dom,
            url=initial_url,
            objective=self.objective,
            completed_tasks=[],
        )

    async def step(
        self, state: BrowserState, action: BrowserAction
    ) -> Tuple[BrowserState, dict]:
        print(f"{Fore.YELLOW}[DEBUG] Executing step with action: {action}")
        new_dom, new_url = await self.execute_browser_action(action)
        current_task = TaskWithActions(
            id=len(state.completed_tasks) + 1,
            description=f"Executed action: {action.action.type}",
            actions_to_be_performed=[action.action],
            result="done",
        )
        new_completed_tasks = state.completed_tasks + [current_task]
        new_state = BrowserState(
            dom=new_dom,
            url=new_url,
            objective=state.objective,
            completed_tasks=new_completed_tasks,
        )
        print(f"{Fore.GREEN}[DEBUG] New state after step - URL: {new_url}")
        return new_state, {}

    async def is_terminal(self, state: BrowserState) -> bool:
        terminal = await is_terminal(state, self.vision)
        print(f"{Fore.CYAN}[DEBUG] Checking if state is terminal: {terminal}")
        return terminal

    async def execute_browser_action(self, action: BrowserAction) -> Tuple[str, str]:
        print(f"{Fore.YELLOW}[DEBUG] Executing browser action: {action.action.type}")

        if action.action.type == ActionType.GOTO_URL:
            print(f"{Fore.CYAN}[DEBUG] Trying to go to url")
            await openurl(url=action.action.website, timeout=action.action.timeout or 0)
            print(f"{Fore.CYAN}[DEBUG] Went to url")
        elif action.action.type == ActionType.TYPE:
            entry = EnterTextEntry(
                query_selector=f"[mmid='{action.action.mmid}']",
                text=action.action.content,
            )
            await entertext(entry)
            await wait_for_navigation()
            print(f"{Fore.CYAN}[DEBUG] Typed text into element")
        elif action.action.type == ActionType.CLICK:
            await click(
                selector=f"[mmid='{action.action.mmid}']",
                wait_before_execution=action.action.wait_before_execution or 0,
            )
            print(f"{Fore.CYAN}[DEBUG] Clicked element")
        elif action.action.type == ActionType.ENTER_TEXT_AND_CLICK:
            await enter_text_and_click(
                text_selector=f"[mmid='{action.action.text_element_mmid}']",
                text_to_enter=action.action.text_to_enter,
                click_selector=f"[mmid='{action.action.click_element_mmid}']",
                wait_before_click_execution=action.action.wait_before_click_execution
                or 0,
            )
            await wait_for_navigation()
            print(f"{Fore.CYAN}[DEBUG] Entered text and clicked element")

        try:
            new_dom = await self.get_current_dom()
        except Exception as e:
            print(f"{Fore.RED}[DEBUG] Error getting DOM after action: {e}")
            new_dom = "Error: Unable to retrieve DOM"

        try:
            new_url = await self.get_current_url()
        except Exception as e:
            print(f"{Fore.RED}[DEBUG] Error getting URL after action: {e}")
            new_url = "Error: Unable to retrieve URL"

        print(f"{Fore.GREEN}[DEBUG] After action execution - New URL: {new_url}")
        return new_dom, new_url

    async def get_current_dom(self) -> str:
        await wait_for_navigation()
        dom = await get_dom_with_content_type(content_type="all_fields")
        print(f"{Fore.CYAN}[DEBUG] Got current DOM (length: {len(dom)})")
        return str(dom)

    async def get_current_url(self) -> str:
        await wait_for_navigation()
        url = await geturl()
        print(f"{Fore.CYAN}[DEBUG] Got current URL: {url}")
        return url


class BrowserMCTSSearchConfig:
    def __init__(self, actor: BaseAgent, critic: BaseAgent, vision: BaseAgent) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.vision = vision

    async def get_actions(self, state: BrowserState) -> List[BrowserAction]:
        print(f"{Fore.YELLOW}[DEBUG] Getting actions for current state")
        actor_input: AgentQActorInput = AgentQActorInput(
            objective=state.objective,
            completed_tasks=state.completed_tasks,
            current_page_dom=state.dom,
            current_page_url=state.url,
        )
        actor_output: AgentQActorOutput = await self.actor.run(actor_input)

        proposed_tasks: List[TaskWithActions] = actor_output.proposed_tasks
        print(f"{Fore.CYAN}[DEBUG] Number of proposed tasks: {len(proposed_tasks)}")

        ranked_actions = await self._rank_actions(state, proposed_tasks)
        print(f"{Fore.CYAN}[DEBUG] Number of sorted actions: {len(ranked_actions)}")

        return ranked_actions

    async def reward(
        self, state: BrowserState, action: BrowserAction, **kwargs
    ) -> Tuple[float, dict]:
        terminal_state = await is_terminal(state=state, vision=self.vision)
        if terminal_state:
            print(f"{Fore.GREEN}[DEBUG] Terminal state reached, reward: 1.0")
            return 1.0, {}
        else:
            print(f"{Fore.RED}[DEBUG] Non-terminal state, reward: -0.01")
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
                    BrowserAction(action=top_task.actions_to_be_performed[0], rank=rank)
                )

                # Remove the top task from remaining tasks
                remaining_tasks = [
                    task for task in remaining_tasks if task.id != top_task.id
                ]
            else:
                print(
                    f"{Fore.MAGENTA}[DEBUG] Warning: No valid top task found in iteration {iteration}. Skipping."
                )

        print(f"{Fore.CYAN}[DEBUG] Sorted actions: {ranked_actions}")
        return ranked_actions


async def is_terminal(state: BrowserState, vision: BaseAgent) -> bool:
    print(f"{Fore.YELLOW}[DEBUG] Checking if state is terminal")
    screenshot = await get_screenshot()
    vision_input: VisionInput = VisionInput(objective=state.objective)
    vision_output: VisionOutput = await vision.run(vision_input, screenshot)
    print(f"{Fore.YELLOW}[DEBUG] Output of vision LLM {vision_output.is_terminal}")
    return vision_output.is_terminal


async def wait_for_navigation(max_retries=3):
    for attempt in range(max_retries):
        try:
            playwright_manager = PlaywrightManager()
            page = await playwright_manager.get_current_page()
            await page.wait_for_load_state("domcontentloaded", timeout=30000)
            print(f"{Fore.GREEN}[DEBUG] Navigation successful on attempt {attempt + 1}")
            return
        except Exception as e:
            print(
                f"{Fore.YELLOW}[DEBUG] Navigation error on attempt {attempt + 1}: {str(e)}"
            )
    print(f"{Fore.RED}[DEBUG] Navigation failed after {max_retries} attempts")


def generate_dpo_pairs(result: MCTSResult):
    dpo_pairs = []
    if result.trace_of_nodes is None or len(result.trace_of_nodes) < 2:
        raise Exception("No valid path found, cannot generate DPO pairs")

    print(f"{Fore.BLUE}[DEBUG] Printing rewards before generating dpo pairs")
    for i in range(len(result.trace_of_nodes)):
        node = result.trace_of_nodes[i]
        print(f"{Fore.BLUE} {node.state.url} - {node.Q}")

    for i in range(len(result.trace_of_nodes) - 1):
        current_node = result.trace_of_nodes[i]
        next_node = result.trace_of_nodes[i + 1]

        if current_node.children:
            winning_action = next_node.action
            for child in current_node.children:
                if child.action != winning_action:
                    dpo_pairs.append((current_node.state, winning_action, child.action))
    return dpo_pairs


def print_result(result: MCTSResult):
    if result.trace is None or len(result.trace) == 0:
        print(f"{Fore.RED}[DEBUG] No valid path found")
        return

    states, actions = result.trace
    print(f"{Fore.GREEN}[DEBUG] Path found:")
    for i, (state, action) in enumerate(zip(states, actions)):
        print(f"{Fore.CYAN}[DEBUG] Step {i}")
        print(f"{Fore.CYAN}[DEBUG]  URL: {state.url}")
        print(f"{Fore.CYAN}[DEBUG]  Action: {action.action.type} - {action}")

    print(f"{Fore.GREEN}[DEBUG] Final URL: {states[-1].url}")
    print(f"{Fore.GREEN}[DEBUG] Cumulative reward: {result.cum_reward}")
    print(f"{Fore.GREEN}[DEBUG] Total steps: {len(actions)}")


def print_dpo_pairs(dpo_pairs):
    if not dpo_pairs:
        print(f"{Fore.RED}No DPO pairs generated.")
        return

    print(f"\n{Fore.MAGENTA}═══════════════ Generated DPO Pairs ═══════════════")

    for i, (state, winning_action, losing_action) in enumerate(dpo_pairs, 1):
        print(f"\n{Fore.CYAN}╔══ Pair {i} ══╗")

        # Print state (URL and trimmed DOM)
        print(f"{Fore.YELLOW}┌─ State ─┐")
        print(f"{Fore.YELLOW}│ URL: {state.url}")
        trimmed_dom = textwrap.shorten(state.dom, width=100, placeholder="...")
        print(f"{Fore.YELLOW}│ DOM: {trimmed_dom}")

        # Print winning action
        print(f"{Fore.GREEN}┌─ Winning Action ─┐")
        print(f"{Fore.GREEN}│ Type: {winning_action.action.type}")
        print(f"{Fore.GREEN}│ Details: {winning_action}")

        # Print losing action
        print(f"{Fore.RED}┌─ Losing Action ─┐")
        print(f"{Fore.RED}│ Type: {losing_action.action.type}")
        print(f"{Fore.RED}│ Details: {losing_action}")

        print(f"{Fore.CYAN}╚{'═' * (len('══ Pair X ══') - 2)}╝")

    print(f"\n{Fore.MAGENTA}═══════════════ End of DPO Pairs ═══════════════")


async def train_loop(
    num_iterations: int = 3,
    output_dir: str = "./dpo_final",
):
    """
    🔁 AgentQ self-training loop (MCTS + DPO)
    - 매 루프마다 policy_model을 메모리 내에서 업데이트
    - 중간 checkpoint 파일 생성 없음
    - 마지막 루프 끝난 후에만 모델 저장
    """

    objective = os.getenv("OBJECTIVE", "")
    model_path = os.getenv("MODEL", "")  # 학습 시작용 모델 (ex: "Qwen/Qwen2.5-7B")

    if not objective:
        raise ValueError("OBJECTIVE environment variable is not set.")

    playwright_manager = PlaywrightManager()
    await playwright_manager.async_initialize()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    policy_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    ref_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    # === Actor, Critic, VisionAgent 초기화 ===
    actor = BaseAgent(
        name="Actor",
        system_prompt="You are the Actor. Propose actions as JSON.",
        input_format=AgentQActorInput,
        output_format=AgentQActorOutput,
        model=policy_model,
        tokenizer=tokenizer,
    )
    critic = BaseAgent(
        name="Critic",
        system_prompt="You are the Critic. Evaluate and rank tasks as JSON.",
        input_format=AgentQCriticInput,
        output_format=AgentQCriticOutput,
        model=policy_model,
        tokenizer=tokenizer,
    )
    vision = BaseVisionAgent(
        name="Vision",
        system_prompt="You are the Vision agent. Decide if terminal state reached, respond in JSON.",
        input_format=VisionInput,
        output_format=VisionOutput,
    )

    world_model = BrowserWorldModel(objective, vision)
    search_config = BrowserMCTSSearchConfig(actor, critic, vision)

    mcts = MCTS(
        n_iters=30,
        w_exp=1.0,
        cum_reward=sum,
        calc_q=np.mean,
        simulate_strategy="max",
        output_strategy="max_reward",
        depth_limit=20,
    )

    last_trainer = None

    # === 학습 루프 ===
    for i in range(num_iterations):
        print(f"\n{Fore.CYAN}========== LOOP {i + 1}/{num_iterations} ==========")

        # (1) MCTS 탐색
        result = await mcts(world_model, search_config)
        print_result(result)

        # (2) DPO pair 생성
        dpo_pairs = generate_dpo_pairs(result)
        print_dpo_pairs(dpo_pairs)

        # (3) Dataset 변환
        def pairs_to_dataset(pairs):
            rows = []
            for state, win, lose in pairs:
                rows.append(
                    {
                        "prompt": textwrap.shorten(
                            state.dom, width=1500, placeholder="..."
                        ),
                        "chosen": getattr(win, "content", str(win)),
                        "rejected": getattr(lose, "content", str(lose)),
                    }
                )
            return Dataset.from_list(rows)

        train_dataset = pairs_to_dataset(dpo_pairs)

        dpo_args = DPOConfig(
            output_dir=None,  # 중간 checkpoint 생성 방지
            save_strategy="no",
            logging_strategy="steps",
            logging_steps=10,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            beta=0.1,
        )

        trainer = DPOTrainer(
            model=policy_model,
            ref_model=ref_model,
            args=dpo_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )

        trainer.train()

        # 🔥 메모리 내에서 최신 모델 반영
        policy_model = trainer.model
        actor.update_model(policy_model, tokenizer)

        # search_config 업데이트
        search_config.actor = actor

        last_trainer = trainer

    # === 마지막 루프 끝난 후 최종 모델 저장 ===
    if last_trainer is not None:
        os.makedirs(output_dir, exist_ok=True)
        last_trainer.save_model(output_dir)
        print(f"{Fore.GREEN}✅ 최종 모델 저장 → {output_dir}")

    await playwright_manager.stop_playwright()


if __name__ == "__main__":
    asyncio.run(train_loop(num_iterations=3))
