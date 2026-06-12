from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router as api_router

app = FastAPI(
    title = "Ajan API Kontrol",
    description = "Çoklu ajan sistemi ile UI arasındaki asenkron iletişim",
    version = "1.0.0"
)

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,  # Prod ortamında bu UI domaini ile sınırlandırılmalı
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

@app.get("/")
async def root():
    return {
        "status": "online", 
        "message": "Çoklu ajan sistemi api katmanı başarılı bir şekilde çalışoyor!",
        "docs_url": "http://127.0.0.1:8000/docs"
    }

app.include_router(api_router, prefix="/api")