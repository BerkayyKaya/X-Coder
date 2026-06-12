import asyncio
import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from src.api.schemas import ChatRequest
from src.api.orchestrator import WorkflowOrchestrator

router = APIRouter()
orchestrator = WorkflowOrchestrator()

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):

    # Veriler text/event-stream ile parça parça UI'a akar
    return StreamingResponse(
        orchestrator.process_stream(request.query),
        media_type="text/event-stream"
    )