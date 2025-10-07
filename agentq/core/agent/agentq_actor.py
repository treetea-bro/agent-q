from datetime import datetime
from string import Template

from transformers import AutoModelForCausalLM, AutoTokenizer

from agentq.core.agent.base import BaseAgent
from agentq.core.memory import ltm
from agentq.core.models.models import AgentQActorInput, AgentQActorOutput
from agentq.core.prompts.prompts import LLM_PROMPTS


class AgentQActor(BaseAgent):
    def __init__(
        self,
        model: AutoModelForCausalLM | None = None,
        tokenizer: AutoTokenizer | None = None,
    ):
        super().__init__(
            name="AgentQActor",
            system_prompt="You are the Actor that proposes next browser tasks.",
            input_format=AgentQActorInput,
            output_format=AgentQActorOutput,
            model=model,
            tokenizer=tokenizer,
        )

    def update_model(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        super().update_model(model, tokenizer)

    @staticmethod
    def __get_ltm():
        return ltm.get_user_ltm()

    def __modify_system_prompt(self, ltm):
        system_prompt: str = LLM_PROMPTS["AGENTQ_ACTOR_PROMPT"]

        substitutions = {
            "basic_user_information": ltm if ltm is not None else "",
        }

        # Use safe_substitute to avoid KeyError
        system_prompt = Template(system_prompt).safe_substitute(substitutions)

        # Add today's day & date to the system prompt
        today = datetime.now()
        today_date = today.strftime("%d/%m/%Y")
        weekday = today.strftime("%A")
        system_prompt += f"\nToday's date is: {today_date}"
        system_prompt += f"\nCurrent weekday is: {weekday}"

        return system_prompt
