import asyncio
import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from src.utils.validators import validate_query
from src.api.schemas import ChatRequest
from src.api.orchestrator import WorkflowOrchestrator

router = APIRouter()
orchestrator = WorkflowOrchestrator()

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):

    validated_query = validate_query(request.query)

    # Veriler text/event-stream ile parça parça UI'a akar
    return StreamingResponse(
        orchestrator.process_stream(validated_query),
        media_type="text/event-stream"
    )