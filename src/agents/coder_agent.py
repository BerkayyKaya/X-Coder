import json
import re
from src.core.model_loader import LLMEngine
from src.agents.base_agent import BaseAgent

# Kullanıcı sorusunu yönlendirecek agent
class CoderAgent(BaseAgent):
    def __init__(self, engine : LLMEngine):
        super().__init__(engine = engine, agent_config_key = "coder_agent")
    
    def generate_code(self, instructions : str):
        raw_response = self._generate(instructions, max_new_tokens = 1100)
        print(f"\n\t[DEBUG] CODER Modelin Ham Çıktısı: {raw_response}\n")

        return raw_response