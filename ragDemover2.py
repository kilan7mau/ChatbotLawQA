import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel
from typing import List, Optional, Any
from dotenv import load_dotenv
import torch
import config
import rag_components
from db.weaviateDB import connect_to_weaviate
from utils.AdvancedLawRetriever import AdvancedLawRetriever
from services.reranker_service import get_reranker_compressor
from schemas.chat import AppState

import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="JuriBot RAG API")


class RAGQueryRequest(BaseModel):
    question: str
    chat_history: Optional[List[Any]] = []


class RAGQueryResponse(BaseModel):
    classification: str
    rewritten_question: str
    answer: str
    sources: Optional[List[Any]] = None


def get_app_state(request: Request):
    """Dependency để lấy app_state từ request"""
    if not hasattr(request.app.state, 'app_state'):
        logger.error("Error in get_app_state: request.app.state.app_state is not set!")
        raise RuntimeError("Application state ('app_state') not found. Initialization failed?")
    return request.app.state.app_state


async def initialize_api_components(app_state: AppState):
    """Khởi tạo các thành phần cần thiết cho API - Đã cải tiến để sử dụng lại rag_components"""
    logger.info("🔸Bắt đầu Khởi tạo API Components")

    load_dotenv()

    # Initialize thread pool executor for parallel processing
    if not hasattr(app_state, 'executor'):
        app_state.__dict__['executor'] = ThreadPoolExecutor(max_workers=4)

    # --- Kết nối tới Weaviate ---
    logger.info("🔸Đang kết nối tới Weaviate...")
    app_state.weaviateDB = connect_to_weaviate(run_diagnostics=False)
    if app_state.weaviateDB is None:
        logger.error("🔸Lỗi kết nối tới Weaviate.")
        raise HTTPException(status_code=500, detail="Lỗi kết nối tới vector database.")

    # --- Lấy Google API Key ---
    app_state.google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not app_state.google_api_key:
        logger.error("🔸Google API Key không được cung cấp.")
        raise HTTPException(status_code=500, detail="Missing Google API Key")

    # --- Xác định thiết bị ---
    app_state.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🔸Sử dụng thiết bị: {app_state.device}")

    # 1. Tải Embedding Model
    logger.info("🔸Đang tải Embedding Model...")
    app_state.embeddings = rag_components.get_huggingface_embeddings(
        config.EMBEDDING_MODEL_NAME, app_state.device
    )
    if not app_state.embeddings:
        logger.error("🔸Lỗi tải embedding model.")
        raise HTTPException(status_code=500, detail="Failed to load embedding model")

    # 2. Tải Vector Store
    logger.info("🔸Đang tải Vector Store...")
    app_state.vectorstore = rag_components.create_or_load_vectorstore(
        embeddings=app_state.embeddings,
        weaviate_url=config.WEAVIATE_URL,
        collection_name=config.WEAVIATE_COLLECTION_NAME,
        weaviate_client=app_state.weaviateDB,
        chunks=None,
    )
    if not app_state.vectorstore:
        logger.error("🔸Lỗi tải vector store.")
        raise HTTPException(status_code=500, detail="Failed to load or create Vectorstore")

    # 3. Tải LLM chính
    logger.info("🔸Đang tải LLM chính...")
    app_state.llm = rag_components.get_google_llm(app_state.google_api_key)
    if not app_state.llm:
        logger.error("🔸Lỗi tải LLM chính.")
        raise HTTPException(status_code=500, detail="Failed to load LLM")
    logger.info("🔸Tải LLM (Google) thành công")

    # 4. Tải reranker
    logger.info("🔸Đang tải reranker...")
    app_state.reranker = get_reranker_compressor()
    if not app_state.reranker:
        logger.error("🔸Lỗi tải reranker.")
        raise HTTPException(status_code=500, detail="Failed to load reranker")

    # 5. Tạo retriever
    logger.info("🔸Đang tạo retriever...")
    app_state.retriever = AdvancedLawRetriever(
        client=app_state.weaviateDB,
        collection_name=config.WEAVIATE_COLLECTION_NAME,
        llm=app_state.llm,
        reranker=app_state.reranker,
        embeddings_model=app_state.embeddings
    )
    if app_state.retriever is None:
        logger.error("🔸Lỗi tạo retriever.")
        raise HTTPException(status_code=500, detail="Failed to create retriever")
    logger.info("🔸Đã tạo retriever thành công.")

    # 6. *** CẢI TIẾN: Sử dụng lại create_qa_chain từ rag_components ***
    logger.info("🔸Đang tạo QA Chain (sử dụng rag_components)...")
    app_state.qa_chain = rag_components.create_qa_chain(
        llm=app_state.llm,
        retriever=app_state.retriever,
        process_input_llm=app_state.llm  # Sử dụng cùng LLM cho preprocessing
    )
    if app_state.qa_chain is None:
        logger.error("🔸Lỗi tạo QA Chain.")
        raise HTTPException(status_code=500, detail="Failed to create QA Chain")

    logger.info("🔸Khởi tạo API Components hoàn tất!")


@app.on_event("startup")
async def startup_event():
    """Khởi tạo ứng dụng khi startup"""
    logger.info("🚀 Starting RAG API initialization...")

    # Tạo AppState instance
    app_state = AppState()

    # Khởi tạo các components
    await initialize_api_components(app_state)

    # Lưu app_state vào request.app.state
    app.state.app_state = app_state

    logger.info("🚀 RAG API initialization completed successfully!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup khi shutdown"""
    logger.info("🛑 Shutting down RAG API...")
    # Cleanup executor if needed
    if hasattr(app.state, 'app_state'):
        app_state = app.state.app_state
        executor = getattr(app_state, 'executor', None)
        if executor:
            executor.shutdown(wait=True)


def run_rag_pipeline(question: str, chat_history: List[Any], qa_chain):
    """
    *** CẢI TIẾN: Đơn giản hóa - chỉ cần 1 lần gọi qa_chain ***
    Vì rag_components.create_qa_chain đã tích hợp sẵn:
    - Unified preprocessing (classification + rewrite)
    - Routing logic
    - Multi-branch execution
    """
    input_data = {"input": question, "chat_history": chat_history}
    return qa_chain.invoke(input_data)


def format_sources_optimized(sources):
    """Optimized source formatting"""
    if not sources:
        return None

    formatted_sources = []
    for doc in sources:
        content = doc.page_content
        preview = content[:200]
        if len(content) > 200:
            preview += "..."

        formatted_sources.append({
            "source": doc.metadata.get("source", "[No source]"),
            "preview": preview
        })

    return formatted_sources


def extract_chain_result(result):
    """
    *** CẢI TIẾN: Trích xuất kết quả từ enhanced chain ***
    Enhanced chain trong rag_components trả về đầy đủ metadata:
    - answer: str
    - context: List[Document] 
    - classification: str
    - rewritten_question: str
    """
    if isinstance(result, dict):
        return {
            "answer": result.get("answer", ""),
            "context": result.get("context", []),
            "classification": result.get("classification", "unknown"),
            "rewritten_question": result.get("rewritten_question", "")
        }
    else:
        # Fallback nếu chain trả về format cũ
        return {
            "answer": str(result),
            "context": [],
            "classification": "unknown",
            "rewritten_question": ""
        }


@app.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest, app_state: AppState = Depends(get_app_state)):
    """
    *** CẢI TIẾN: RAG query endpoint đơn giản hóa ***
    Không còn cần parallel processing phức tạp vì rag_components đã tối ưu
    """

    if not app_state.qa_chain:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized.")

    try:
        chat_history = request.chat_history or []
        total_start = time.time()

        # *** SINGLE CALL: Sử dụng qa_chain đã tích hợp sẵn preprocessing + routing ***
        loop = asyncio.get_event_loop()
        executor = getattr(app_state, 'executor', None)
        if not executor:
            raise HTTPException(status_code=500, detail="Thread pool executor not initialized.")

        rag_result = await loop.run_in_executor(
            executor, run_rag_pipeline, request.question, chat_history, app_state.qa_chain
        )

        # Trích xuất kết quả
        extracted = extract_chain_result(rag_result)

        # Format sources
        formatted_sources = format_sources_optimized(extracted["context"])

        response = RAGQueryResponse(
            classification=extracted["classification"],
            rewritten_question=extracted["rewritten_question"],
            answer=extracted["answer"],
            sources=formatted_sources
        )

        total_time = time.time() - total_start
        logger.info(f"[RAG API] Single chain execution time: {total_time:.3f}s")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in RAG pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Error during RAG pipeline: {e}")


@app.get("/health")
async def health_check(app_state: AppState = Depends(get_app_state)):
    """Health check endpoint"""
    return {
        "status": "healthy",
        "components": {
            "llm": getattr(app_state, 'llm', None) is not None,
            "retriever": getattr(app_state, 'retriever', None) is not None,
            "qa_chain": getattr(app_state, 'qa_chain', None) is not None,
            "embeddings": getattr(app_state, 'embeddings', None) is not None,
            "weaviateDB": getattr(app_state, 'weaviateDB', None) is not None,
            "reranker": getattr(app_state, 'reranker', None) is not None
        }
    }


@app.post("/warmup")
async def warmup(app_state: AppState = Depends(get_app_state)):
    """Warmup endpoint to ensure models are loaded"""
    try:
        test_query = "Luật giao thông"
        test_request = RAGQueryRequest(question=test_query, chat_history=[])

        # Run a quick test
        await rag_query(test_request, app_state)
        return {"status": "warmed up successfully"}
    except Exception as e:
        logger.warning(f"Warmup failed: {e}")
        return {"status": "warmup completed with warnings", "warning": str(e)}