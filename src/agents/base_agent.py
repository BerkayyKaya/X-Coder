import os

import yaml

from src.core.model_loader import LLMEngine


# Bütün agentlar için base sınıf
class BaseAgent:
    def __init__(self, engine : LLMEngine, agent_config_key : str):
        self.agent = engine
        self.agent_config_key = agent_config_key
        self.prompts = self._load_prompts()
        self.instructions = self.prompts.get(self.agent_config_key, {}).get("instructions", "")

    def _load_prompts(self):
        prompt_path = os.getenv("PROMPTS_YAML_PATH", "./config/prompts.yaml")

        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"\n\t[HATA] Prompt dosyası bulunamadı: {prompt_path}")
        
        with open(prompt_path, "r", encoding = "utf-8") as file:
            return yaml.safe_load(file)
        
    def _generate(self, user_query : str, max_new_tokens : int = 150, **kwargs) -> str:
        messages = [
            {"role" : "system", "content" : self.instructions},
            {"role" : "user", "content" : user_query}
        ]

        temperature = kwargs.get("temperature", None)
        if self.agent.backend == "transformers":
            if temperature == 0.0 or temperature is None:
                kwargs["do_sample"] = False
                kwargs.pop("temperature", None) 
            else:
                kwargs["do_sample"] = True
                kwargs["temperature"] = temperature
                
        elif self.agent.backend == "llama-cpp":
            kwargs["temperature"] = max(0.0, temperature) # Negatif olmamalı

        return self.agent.generate_response(messages, max_new_tokens = max_new_tokens, **kwargs)
    
    