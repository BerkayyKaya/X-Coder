from src.core.model_loader import LLMEngine
from src.agents.base_agent import BaseAgent

class ChatterAgent(BaseAgent):
    def __init__(self, engine: LLMEngine):
        super().__init__(engine = engine, agent_config_key = "chatter_agent")
    
    def generate_response(self, context : dict) -> str:

        instructions_dict = self.instructions

        if context.get("code") is not None:
            raw_system_prompt = instructions_dict.get("presentation_role", "")
            formatted_system_prompt = raw_system_prompt.replace("{plan}", str(context.get("plan", "")))
        else:
            raw_system_prompt = instructions_dict.get("mentor_role", "")
            formatted_system_prompt = raw_system_prompt 

        self.instructions = formatted_system_prompt

        raw_response = self._generate(user_query = context.get("query", ""), max_new_tokens = 2048, do_sample = True, temperature = 0.2)
        print(f"\n\t[DEBUG] CHATTER Modelin Ham Çıktısı: {raw_response}\n")

        return raw_response
