import json
import re
from src.core.model_loader import LLMEngine
from src.agents.base_agent import BaseAgent

# Kullanıcı sorusunu yönlendirecek agent
class RouterAgent(BaseAgent):
    def __init__(self, engine : LLMEngine):
        super().__init__(engine = engine, agent_config_key = "router_agent")
    
    def route_request(self, user_query : str):
        raw_response = self._generate(user_query, do_sample = False) # do_sample False ise temperature verilmez
        print(f"\n\t[DEBUG] ROUTER Modelin Ham Çıktısı: {raw_response}\n")

        return self._extract_json(raw_response)
    
    def _extract_json(self, text):
        try:
            # Süslü parantezler içindeki her şeyi yakalayan Regex
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
            else:
                return {"route": "UNKNOWN", "error": "JSON formatı bulunamadı"}
        except json.JSONDecodeError:
            return {"route": "UNKNOWN", "error": "Bozuk JSON formatı"}