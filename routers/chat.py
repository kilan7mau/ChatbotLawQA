from fastapi import APIRouter, HTTPException, Request
from schemas.chat import QueryRequest, AnswerResponse
from services.chat_service import ask_question_service, stream_chat_generator
from dependencies import get_app_state
import logging
from fastapi.responses import StreamingResponse

# Thiết lập logger
logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("", response_model=AnswerResponse)
async def chat_message(request_body: QueryRequest, request: Request):
    """Core chat endpoint - removed auth dependency"""
    app_state = get_app_state(request=request)
    result = await ask_question_service(app_state, request_body)

    if not result:
        raise HTTPException(status_code=500, detail="Error during QA Chain invocation")
    return result


@router.get("/stream")
async def stream_chat_endpoint(
    chat_id: str,  # Lấy từ query param
    input: str,    # Lấy từ query param
    request: Request,
    chat_history: str = "[]"  # Optional chat history as JSON string
):
    """Streaming endpoint - removed auth dependency, added chat_history support"""
    app_state = get_app_state(request=request)

    # Kiểm tra input cơ bản
    if not chat_id or not input:
        raise HTTPException(status_code=400, detail="chat_id and input are required.")

    # Parse chat_history from JSON string
    try:
        import json
        chat_history_list = json.loads(chat_history) if chat_history != "[]" else []
    except json.JSONDecodeError:
        chat_history_list = []

    return StreamingResponse(
        stream_chat_generator(app_state, chat_id, input, chat_history_list),
        media_type="text/event-stream"
    )


@router.post("/rag/query")
async def rag_query_endpoint(request_body: QueryRequest, request: Request):
    """
    Core RAG endpoint - equivalent to demover2.py /rag/query
    Direct RAG query without session management
    """
    app_state = get_app_state(request=request)
    
    try:
        result = await ask_question_service(app_state, request_body)
        if not result:
            raise HTTPException(status_code=500, detail="Error during RAG pipeline")
        return result
    except Exception as e:
        logger.error(f"Error in RAG query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {str(e)[:100]}")


@router.get("/health")
async def health_check(request: Request):
    """
    Health check endpoint - equivalent to demover2.py /health
    """
    app_state = get_app_state(request=request)
    
    return {
        "status": "healthy",
        "components": {
            "llm": app_state.llm is not None,
            "retriever": app_state.retriever is not None,
            "qa_chain": app_state.qa_chain is not None,
            "embeddings": app_state.embeddings is not None,
            "weaviateDB": app_state.weaviateDB is not None,
            "reranker": app_state.reranker is not None
        }
    }


@router.post("/warmup")
async def warmup_endpoint(request: Request):
    """
    Warmup endpoint - equivalent to demover2.py /warmup
    Preload models with a test query
    """
    app_state = get_app_state(request=request)
    
    try:
        test_query = QueryRequest(chat_id="warmup", input="Luật giao thông")
        await ask_question_service(app_state, test_query)
        return {"status": "warmed up successfully"}
    except Exception as e:
        logger.warning(f"Warmup failed: {e}")
        return {"status": "warmup completed with warnings", "warning": str(e)}