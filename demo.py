import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any
import config
import rag_components
from db.weaviateDB import connect_to_weaviate
from utils.AdvancedLawRetriever import AdvancedLawRetriever
from services.reranker_service import get_reranker_compressor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from prompt_templete import UNIFIED_PREPROCESSING_PROMPT

app = FastAPI(title="LexiBot RAG API")

class RAGQueryRequest(BaseModel):
    question: str
    chat_history: Optional[List[Any]] = []

class RAGQueryResponse(BaseModel):
    classification: str
    rewritten_question: str
    answer: str
    sources: Optional[List[Any]] = None

# Global objects for reuse
embeddings = None
weaviate_client = None
llm = None
reranker = None
retriever = None
qa_chain = None

@app.on_event("startup")
def startup_event():
    global embeddings, weaviate_client, llm, reranker, retriever, qa_chain
    # 1. Load embedding model
    embeddings = rag_components.get_huggingface_embeddings(config.EMBEDDING_MODEL_NAME, device="cpu")
    if not embeddings:
        raise RuntimeError("Failed to load embedding model.")
    # 2. Connect to Weaviate
    weaviate_client = connect_to_weaviate(run_diagnostics=False)
    if not weaviate_client:
        raise RuntimeError("Failed to connect to Weaviate.")
    # 3. Load LLM
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if hasattr(rag_components, 'get_google_llm') and google_api_key:
        llm = rag_components.get_google_llm(google_api_key)
    if not llm:
        raise RuntimeError("Failed to load LLM. Please set GOOGLE_API_KEY.")
    # 4. Load reranker
    reranker = get_reranker_compressor()
    if not reranker:
        raise RuntimeError("Failed to load reranker.")
    # 5. Instantiate retriever
    retriever = AdvancedLawRetriever(
        client=weaviate_client,
        collection_name=config.WEAVIATE_COLLECTION_NAME,
        llm=llm,
        reranker=reranker,
        embeddings_model=embeddings
    )
    # 6. Build the RAG chain
    qa_chain = rag_components.create_qa_chain(
        llm=llm,
        retriever=retriever,
        process_input_llm=llm
    )

@app.post("/rag/query", response_model=RAGQueryResponse)
def rag_query(request: RAGQueryRequest):
    global qa_chain, llm
    if not qa_chain or not llm:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized.")
    try:
        # Preprocessing: classification & rewritten question
        preprocessing_prompt = ChatPromptTemplate.from_template(UNIFIED_PREPROCESSING_PROMPT)
        parser = JsonOutputParser()
        preprocessing_chain = preprocessing_prompt | llm | parser
        preprocessing_result = preprocessing_chain.invoke({"input": request.question, "chat_history": request.chat_history or []})
        classification = preprocessing_result.get("classification", "unknown")
        rewritten_question = preprocessing_result.get("rewritten_question", request.question)
        # RAG pipeline
        input_data = {"input": request.question, "chat_history": request.chat_history or []}
        result = qa_chain.invoke(input_data)
        answer = result["answer"] if isinstance(result, dict) and "answer" in result else str(result)
        sources = result.get("context") if isinstance(result, dict) else None
        # Format sources for output (only show metadata and preview)
        formatted_sources = None
        if sources:
            formatted_sources = [
                {
                    "source": doc.metadata.get("source", "[No source]"),
                    "preview": doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else "")
                } for doc in sources
            ]
        return RAGQueryResponse(
            #thay đổi các biến để hiện thị rõ ràng hơn
            classification=classification,
            rewritten_question=rewritten_question,
            answer=answer,
            sources=formatted_sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during RAG pipeline: {e}")

