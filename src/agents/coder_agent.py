import json
import re
from src.core.model_loader import LLMEngine
from src.agents.base_agent import BaseAgent

# Kullanıcı sorusunu yönlendirecek agent
class CoderAgent(BaseAgent):
    def __init__(self, engine : LLMEngine):
        super().__init__(engine = engine, agent_config_key = "coder_agent")
    
    def generate_code(self, instructions : str):
        raw_response = self._generate(instructions, max_new_tokens = 2048, do_sample = False, repetition_penalty = 1.15) # do_sample False ise temperature verilmez
        print(f"\n\t[DEBUG] CODER Modelin Ham Çıktısı: {raw_response}\n")

        return self._parse_output(raw_output = raw_response)
    
    def _parse_output(self, raw_output : str):
        parsed_output = {
            "code" : "",
            "explanation" : ""
        }

        code_match = re.search(r"```.*?```", raw_output, re.DOTALL)
        if code_match:
            parsed_output["code"] = code_match.group().strip()
        else:
            parsed_output["code"] = raw_output
        
        json_match = re.search(r"```json\n(.*?)```", raw_output, re.DOTALL | re.IGNORECASE)
        if json_match:
            try:
                json_data = json.loads(json_match.group(1).strip())
                parsed_output["explanation"] = json_data.get("explanation", "")
            except json.JSONDecodeError:
                exp_match = re.search(r'"explanation"\s*:\s*"(.*?)"', json_match.group(1), re.DOTALL)
                if exp_match:
                    parsed_output["explanation"] = exp_match.group(1).strip()
        return parsed_output