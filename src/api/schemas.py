from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    # Kullanıcının göndereceği ana metin
    query: str = Field(..., description="Kullanıcının sisteme sorduğu soru veya komut")