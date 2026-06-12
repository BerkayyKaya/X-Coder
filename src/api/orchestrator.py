import json
import asyncio
import os
import time
from dotenv import load_dotenv
from src.core.model_loader import LLMEngine
from src.agents.router_agent import RouterAgent
from src.agents.planner_agent import PlannerAgent
from src.agents.coder_agent import CoderAgent

load_dotenv("./config/.env")

class WorkflowOrchestrator:
    def __init__(self):
        print("\t[INFO] orchestrator başlatılıyor...")

        self.current_engine = None
        self.current_model_path = None

        print("\t[INFO] orchestrator hazır!")

    def _get_or_load_model(self, model_path : str, tokenizer_path : str, quantization : bool = False):
        
        if self.current_model_path == model_path and self.current_engine is not None:
            return self.current_engine
        
        if self.current_engine is not None:
            self.current_engine.unload_model()
        
        self.current_engine = LLMEngine(
            model_path = model_path,
            tokenizer_path = tokenizer_path,
            quantizaton = quantization
        )

        self.current_model_path = model_path

        return self.current_engine


    async def process_stream(self, user_query : str):
        """
        FastAPI'nin StreamingResponse yapısına uyumlu generator fonksiyonu.
        Her yield kelimesi, Next.js'e anlık bir paket gönderir.
        """
        context = {
            "query" : user_query,
            "plan" : None,
            "code" : None,
            "tester_feedback" : None,
            "coder_attemps" : 0
        }

        current_step = "ROUTER"

        while True:
            if current_step == "ROUTER":
                yield f"data: {json.dumps({'status' : 'routing', 'message' : 'Yönlendirici isteği inceliyor...'})}\n"
                start_time = time.time()


                engine = self._get_or_load_model(
                    model_path = os.getenv("ROUTER_PATH"),
                    tokenizer_path = os.getenv("ROUTER_TOKENIZER_PATH"),
                    quantization = True
                )

                router_agent = RouterAgent(engine = engine)

                routing_decision = router_agent.route_request(context["query"])

                end_time = time.time()

                print(f"\n\t[INFO] ROUTER Model Cevap Verme Süresi: {end_time - start_time} saniye.")

                current_step = routing_decision.get("route", "UNKNOWN")
            
            elif current_step == "PLANNER":
                yield f"data: {json.dumps({'status' : 'planning', 'message' : 'Kullanıcı isteğine göre plan yapılıyor...'})}\n"
                
                start_time = time.time()

                engine = self._get_or_load_model(
                    model_path = os.getenv("ROUTER_PATH"),
                    tokenizer_path = os.getenv("ROUTER_TOKENIZER_PATH"),
                    quantization = True
                )

                planner_agent = PlannerAgent(engine = engine)

                context["plan"] = planner_agent.create_plan(context["query"])
                
                end_time = time.time()

                print(f"\n\t[INFO] PLANNER Model Cevap Verme Süresi: {end_time - start_time} saniye.")

                current_step = "CODER"

            elif current_step == "CODER":
                yield f"data: {json.dumps({'status' : 'coding', 'message' : 'Kod yazılıyor...'})}\n"

                start_time = time.time()

                engine = self._get_or_load_model(
                    model_path = os.getenv("CODER_PATH"),
                    tokenizer_path = os.getenv("CODER_TOKENIZER_PATH"),
                    quantization = True
                )

                coder_agent = CoderAgent(engine = engine)

                if context["plan"] is not None:
                    context["code"] = coder_agent.generate_code(instructions = context["plan"])
                else:
                    context["code"] = coder_agent.generate_code(instructions = context["query"])

                end_time = time.time()

                print(f"\n\t[INFO] CODER Model Cevap Verme Süresi: {end_time - start_time} saniye.")

                context["coder_attemps"] += 1

                print(f"\n\t[SON ÇIKTI] {context}")
                
                current_step = "TESTER"
            
            elif current_step == "TESTER":
                yield f"data: {json.dumps({'status' : 'testing', 'message' : 'Kod test ediliyor...'})}\n"
                is_valid = True # kod doğru mu?

                if not is_valid and context["coder_attemps"] < 3:
                    context["tester_feedback"] = "Hata: Satır 23 Null pointer exception bulundu."
                    current_step = "CODER"
                else:
                    current_step = "CHAT"
            
            elif current_step == "CHAT":
                yield f"data: {json.dumps({'status' : 'chat', 'message' : 'Sonuç hazırlanıyor...'})}\n"
                # chatterin cevabı
                current_step = "END"
            
            if current_step == "END":
                yield f"data: {json.dumps({'status' : 'done', 'message' : 'İşlem tamamlandı!'})}\n"
                break