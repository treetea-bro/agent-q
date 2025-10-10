from enum import Enum, IntEnum
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ==============================
# Enums
# ==============================
class State(str, Enum):
    PLAN = "plan"
    BROWSE = "browse"
    COMPLETED = "completed"
    AGENTQ_BASE = "agentq_base"
    AGENTQ_ACTOR = "agentq_actor"
    AGENTQ_CRITIC = "agentq_critic"


class ActionType(str, Enum):
    CLICK = "CLICK"
    TYPE = "TYPE"
    GOTO_URL = "GOTO_URL"
    ENTER_TEXT_AND_CLICK = "ENTER_TEXT_AND_CLICK"
    SOLVE_CAPTCHA = "SOLVE_CAPTCHA"


# ==============================
# Action Types
# ==============================
class ClickAction(BaseModel):
    type: Literal[ActionType.CLICK] = Field(description="Click an element by mmid.")
    mmid: int = Field(description="The mmid of the element to click.")
    wait_before_execution: Optional[float] = Field(
        default=None, description="Wait time before executing click."
    )


class TypeAction(BaseModel):
    type: Literal[ActionType.TYPE] = Field(
        description="Type text into an element by mmid."
    )
    mmid: int = Field(description="The mmid of the element to type into.")
    content: str = Field(description="The text to type into the element.")


class GotoAction(BaseModel):
    type: Literal[ActionType.GOTO_URL] = Field(description="Navigate to a given URL.")
    website: str = Field(description="Target URL (must include http/https).")
    timeout: Optional[float] = Field(
        default=None, description="Optional wait time after load."
    )


class EnterTextAndClickAction(BaseModel):
    type: Literal[ActionType.ENTER_TEXT_AND_CLICK] = Field(
        description="Enter text and click another element."
    )
    text_element_mmid: int = Field(description="The mmid of the text field.")
    text_to_enter: str = Field(description="The text to enter.")
    click_element_mmid: int = Field(description="The mmid of the button to click.")
    wait_before_click_execution: Optional[float] = Field(
        default=None, description="Wait time before click execution."
    )


class SolveCaptcha(BaseModel):
    type: Literal[ActionType.SOLVE_CAPTCHA] = Field(
        description="Solve a captcha and click submit."
    )
    text_element_mmid: int = Field(description="The mmid of the captcha input element.")
    click_element_mmid: int = Field(description="The mmid of the submit button.")
    wait_before_click_execution: Optional[float] = Field(
        default=None, description="Wait time before executing captcha click."
    )


class Score(IntEnum):
    FAIL = 0
    PASS = 1


Action = Union[
    ClickAction,
    TypeAction,
    GotoAction,
    EnterTextAndClickAction,
    SolveCaptcha,
]


# ==============================
# Task Models
# ==============================
class Task(BaseModel):
    id: int
    description: str
    url: Optional[str] = None
    result: Optional[str] = None


class TaskWithActions(BaseModel):
    id: int
    description: str
    actions_to_be_performed: Optional[List[Action]] = None
    result: Optional[str] = None


# ==============================
# Memory / Planner / Agents
# ==============================
class Memory(BaseModel):
    objective: str
    current_state: State
    plan: Optional[Union[List[Task], List[TaskWithActions]]] = None
    thought: str
    completed_tasks: Optional[Union[List[Task], List[TaskWithActions]]] = None
    current_task: Optional[Union[Task, TaskWithActions]] = None
    final_response: Optional[str] = None
    current_tasks_for_eval: Optional[List[TaskWithActions]] = None
    sorted_tasks: Optional[List[TaskWithActions]] = None

    class Config:
        use_enum_values = True


class PlannerInput(BaseModel):
    objective: str
    completed_tasks: Optional[List[Task]] = None
    task_for_review: Optional[Task] = None


class PlannerOutput(BaseModel):
    plan: Optional[List[Task]] = None
    thought: str
    next_task: Optional[Task] = None
    is_complete: bool
    final_response: Optional[str] = None


# ==============================
# Browser Executor
# ==============================
class BrowserNavInput(BaseModel):
    task: Task


class BrowserNavOutput(BaseModel):
    completed_task: Task


# ==============================
# AgentQ Base / Actor / Critic
# ==============================
class AgentQBaseInput(BaseModel):
    objective: str
    completed_tasks: Optional[List[Task]] = None
    current_page_url: str
    current_page_dom: str


class AgentQBaseOutput(BaseModel):
    thought: str
    plan: List[Task]
    next_task: Optional[Task] = None
    next_task_actions: Optional[List[Action]] = None
    is_complete: bool
    final_response: Optional[str] = None


class AgentQActorInput(BaseModel):
    objective: str
    completed_tasks: Optional[List[TaskWithActions]] = None
    current_page_url: str
    current_page_dom: str


class AgentQActorOutput(BaseModel):
    thought: str
    proposed_tasks: Optional[List[TaskWithActions]] = None
    is_complete: bool
    final_response: Optional[str] = None


class AgentQCriticInput(BaseModel):
    objective: str
    completed_tasks: Optional[List[TaskWithActions]] = None
    tasks_for_eval: List[TaskWithActions]
    current_page_url: str
    current_page_dom: str


class AgentQCriticOutput(BaseModel):
    thought: str
    top_task: TaskWithActions


# ==============================
# Vision / Eval / Captcha
# ==============================
class VisionInput(BaseModel):
    objective: str


class VisionOutput(BaseModel):
    is_terminal: bool


class EvalAgentInput(BaseModel):
    objective: str
    agent_output: str
    current_page_url: str
    current_page_dom: str


class EvalAgentOutput(BaseModel):
    score: Score


class CaptchaAgentInput(BaseModel):
    objective: str


class CaptchaAgentOutput(BaseModel):
    captcha: str
    success: bool


# ==============================
# Monte Carlo / DPO
# ==============================
class BrowserState(BaseModel):
    dom: str
    url: str
    objective: str
    completed_tasks: Optional[List[TaskWithActions]] = None


class BrowserAction(BaseModel):
    task_with_action: TaskWithActions
    rank: float = Field(description="Higher rank = better action.")


class DPOState(BaseModel):
    objective: str
    dom: str


class DPOAction(BaseModel):
    description: str
    action: Action


class DPOPair(BaseModel):
    state: DPOState
    winning_action: DPOAction
    losing_action: DPOAction
