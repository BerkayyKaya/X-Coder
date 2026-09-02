import json
import re

from llama_cpp import LlamaGrammar

from src.agents.base_agent import BaseAgent
from src.core.model_loader import LLMEngine


# Kullanıcı sorusunu yönlendirecek agent
class RouterAgent(BaseAgent):
    def __init__(self, engine : LLMEngine):
        super().__init__(engine = engine, agent_config_key = "router_agent")

        grammar_text = r'''
        root ::= "{" ws "\"route\"" ws ":" ws "\"" agent "\"" ws "}"
        agent ::= "PLANNER" | "CODER" | "CHAT"
        ws ::= [ \t\n]*
        '''
        self.grammar = LlamaGrammar.from_string(grammar_text)
    
    def route_request(self, user_query : str):
        raw_response = self._generate(user_query, temperature = 0.0)
        print(f"\n\t[DEBUG] ROUTER Modelin Ham Çıktısı: {raw_response}\n")

        return self._extract_json(raw_response)

    def _generate(self, user_query: str, max_new_tokens: int = 15, **kwargs) -> str:
        # grammar objesini kwargs içine ekliyoruz
        if self.agent.backend == "transformers":
            pass
        elif self.agent.backend == "llama-cpp":
            kwargs["grammar"] = self.grammar
        
        # Üst sınıfın (BaseAgent) _generate metodunu çağırıyoruz.
        # Böylece grammar objesi **kwargs üzerinden model_loader.py'deki 
        # create_chat_completion fonksiyonuna kadar taşınacak.
        return super()._generate(user_query, max_new_tokens = max_new_tokens, **kwargs)
    
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