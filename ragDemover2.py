import os
from dotenv import load_dotenv
import config
from schemas.chat import AppState
from langchain_groq import ChatGroq
from utils.AdvancedLawRetriever import AdvancedLawRetriever
from services.reranker_service import get_reranker_compressor
from db.weaviateDB import connect_to_weaviate
import torch
import rag_components
import logging

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()  # Gửi log đến stdout
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

def get_app_state(request: Request):
    if not hasattr(request.app.state, 'app_state'):
        print("Error in get_app_state: request.app.state.app_state is not set!")
        raise RuntimeError("Application state ('app_state') not found. Initialization failed?")
    return request.app.state.app_state



async def initialize_api_components(app_state: AppState):
    """Khởi tạo các thành phần cần thiết cho API """
    logger.info("🔸Bắt đầu Khởi tạo API Components")

    load_dotenv()
    # --- Kiểm tra kết nối tới Redis ---
    app_state.process_input_llm = ChatGroq(model=config.GROQ_MODEL_NAME,temperature=0.2)
    try:
        app_state.redis = await get_redis_client() # Gọi hàm khởi tạo redis
    except Exception as e:
        logger.error(f"☠️ LỖI NGHIÊM TRỌNG khi khởi tạo Redis trong initialize_api_components: {e}")
        raise
    app_state.dict = load_legal_dictionary(config.LEGAL_DIC_FOLDER+ "/legal_terms.json")
    app_state.weaviateDB = connect_to_weaviate(run_diagnostics=False)
    # --- Kiểm tra kết nối tới MongoDB ---
    if user_collection is  None or app_state.weaviateDB is None:
        logger.error("🔸Lỗi kết nối tới MongoDB hoặc Weaviate.")
        raise HTTPException(status_code=500, detail="Lỗi kết nối tới database.")

    app_state.google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not app_state.google_api_key:
        logger.error("🔸GG API Key không được cung cấp.")
        raise HTTPException(status_code=500, detail="Missing GG API Key")

    app_state.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🔸Sử dụng thiết bị: {app_state.device}")

    # 1. Tải Embedding Model (giữ nguyên)
    print(f"Đang tải Embedding Model...")
    app_state.embeddings = rag_components.get_huggingface_embeddings(
        EMBEDDING_MODEL_NAME, app_state.device
    )
    if not app_state.embeddings:
        raise HTTPException(status_code=500, detail="Failed to load embedding model")

    # 2. Tải Vector Store
    print(f"Đang tải Vector Store...")
    app_state.vectorstore = rag_components.create_or_load_vectorstore(
        embeddings=app_state.embeddings,
        weaviate_url=WEAVIATE_URL,
        collection_name=WEAVIATE_COLLECTION_NAME,
        weaviate_client=app_state.weaviateDB,
        chunks=None,
    )

    if not app_state.vectorstore:
         raise HTTPException(status_code=500, detail="Failed to load or create Vectorstore")

    # 3. Tải LLM
    logger.info(f"🔸Đang tải LLM...")

    llm = rag_components.get_google_llm(app_state.google_api_key)
    app_state.llm = llm
    logger.info(f"🔸Tải LLM (Groq) thanh cong")

    if not app_state.llm:
        raise HTTPException(status_code=500, detail="Failed to load LLM")

    # 4. Tạo retriever (giữ nguyên)
    logger.info(f"🔸Đang tạo retriever...")

    app_state.reranker = get_reranker_compressor() # Singleton re-ranker

    app_state.retriever = AdvancedLawRetriever(
        client=app_state.weaviateDB,
        collection_name=WEAVIATE_COLLECTION_NAME,
        llm=app_state.llm,
        reranker=app_state.reranker, # Singleton re-ranker
        embeddings_model=app_state.embeddings
    )

    if app_state.retriever is None:
        raise HTTPException(status_code=500, detail="Failed to create retriever")
    logger.info(f"🔸Đã tạo retriever thành công.")

    # 5. Tạo QA Chain (giữ nguyên)
    logger.info(f"🔸Đang tạo QA Chain...")

    app_state.qa_chain = rag_components.create_qa_chain(
        llm=app_state.llm,
        retriever=app_state.retriever,
        process_input_llm=app_state.process_input_llm
    )
    if app_state.qa_chain is None:
        raise HTTPException(status_code=500, detail="Failed to create QA Chain")

    logger.info(f"🔸Khởi tạo API Components hoàn tất ")