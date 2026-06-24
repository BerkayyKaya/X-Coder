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

        return self._parse_output(text = raw_response)

    def _parse_output(self, text : str):
        banned_words = ["Absolutely", "happy", "Here's", "Certainly!"]
        codes = re.findall(r"(```.*?```)", text, re.DOTALL)
        texts = re.findall(r"(.*?)```(?:.*?)```", text, re.DOTALL)

        parsed_output = {
            "code": "",
            "explanation" : ""
        }

        if len(codes) > 0 and len(texts) > 0:
            code_dict = {}
            for i, block in enumerate(codes):
                code_dict.update({i+1: f"{block}"})
            parsed_output["code"] = code_dict
            del code_dict

            text_dict = {}
            for i, block in enumerate(texts):
                if any(word.lower() in block.lower() for word in banned_words) and i == 0:
                    index = block.find("\n\n")
                    if index != -1:
                        block = block[index + 1: ]

                text_dict.update({i+1: f"{block}"})
            parsed_output["explanation"] = text_dict
            del text_dict
        else:
            code_dict = {}
            for i, block in enumerate(codes):
                code_dict.update({i+1: f"{block}"})
            parsed_output["code"] = code_dict
            del code_dict
            parsed_output["explanation"] = "No explanation found."

        return parsed_output