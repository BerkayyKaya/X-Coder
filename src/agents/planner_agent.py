import json
import re
from src.core.model_loader import LLMEngine
from src.agents.base_agent import BaseAgent

# Kullanıcı sorusunu yönlendirecek agent
class PlannerAgent(BaseAgent):
    def __init__(self, engine : LLMEngine):
        super().__init__(engine = engine, agent_config_key = "planner_agent")
    
    def create_plan(self, user_query : str):
        raw_response = self._generate(user_query, max_new_tokens = 1024, do_sample = True, temperature = 0.1)
        print(f"\n\t[DEBUG] PLANNER Modelin Ham Çıktısı: {raw_response}\n")

        return raw_response