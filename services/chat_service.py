# BACKUP OF ORIGINAL COMPLEX FUNCTIONS - COMMENTED OUT
# Original functions with Redis/MongoDB/Auth dependencies are preserved below for reference
from fastapi import Depends, HTTPException
from schemas.chat import QueryRequest, AnswerResponse, SourceDocument
#from schemas.user import UserOut
import time
import json
from utils.utils import search_term_in_dictionary, minimal_preprocess_for_llm
import os
import logging
from datetime import datetime, timezone
import asyncio
logger = logging.getLogger(__name__)


async def ask_question_service(app_state, request: QueryRequest):
    """
    Refactored RAG query service - removed auth/MongoDB/Redis dependencies
    """
    chat_id = request.chat_id
    question_content = request.input  # Giữ lại câu hỏi gốc của user
    
    start_time = time.time()
    current_utc_time = datetime.now(timezone.utc)  # Giữ lại cho consistency
    
    # --- 1. Tiền xử lý câu hỏi ---
    cleaned_question = minimal_preprocess_for_llm(question_content)

    # --- 2. Kiểm tra từ điển thuật ngữ (nếu có) ---
    if hasattr(app_state, 'dict') and app_state.dict:
        term_result = search_term_in_dictionary(cleaned_question, app_state.dict)
        if term_result:
            answer_def = term_result.get("definition", "Không thể tìm thấy định nghĩa.")
            friendly_answer = f"Xin chào! Về câu hỏi '{question_content}' của bạn, tôi đã tìm thấy thông tin sau:\n\n{answer_def}\n\nHy vọng thông tin này hữu ích cho bạn. Bạn có muốn tìm hiểu thêm về chủ đề này hoặc có câu hỏi nào khác không? 😊"
            return AnswerResponse(
                answer=friendly_answer,
                sources=[
                    SourceDocument(
                        source="Thuật ngữ pháp lý",
                        page_content_preview=f"Định nghĩa thuật ngữ từ cơ sở dữ liệu"
                    )
                ],
                processing_time=round(time.time() - start_time, 2)
            )

    # --- 3. Kiểm tra QA Chain ---
    if not app_state.qa_chain:
        logger.error("QA Chain chưa được khởi tạo.")
        raise HTTPException(status_code=503, detail="Service Unavailable: QA Chain not ready.")

    # --- 4. Chuẩn bị chat history và input cho QA Chain ---
    try:
        # Tạo chat history từ request hoặc sử dụng empty list
        chat_history = getattr(request, 'chat_history', []) or []
        
        # Format chat history cho prompt
        chat_history_string = format_chat_history_for_prompt(chat_history)
        
        input_data_for_chain = {
            "chat_history": chat_history_string,
            "input": cleaned_question
        }

    except Exception as e:
        logger.error(f"Lỗi khi chuẩn bị chat history cho Langchain (chat_id: {chat_id}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Lỗi xử lý lịch sử chat.")


    # --- 5. Gọi QA Chain ---
    try:
        logger.debug(f"Input to QA Chain (chat_id: {chat_id}): {input_data_for_chain}")

        # Metadata cho LangSmith trace
        langsmith_metadata = {
            "chat_id": chat_id,
            "original_question": question_content,
            "cleaned_question": cleaned_question,
            "request_id": getattr(request, 'request_id', "N/A")
        }

        chain_result = app_state.qa_chain.invoke(input_data_for_chain, config={
                    "metadata": langsmith_metadata,
                    "run_name": f"AskService_QA_Invoke_ChatID_{chat_id[:8]}"
                })

        # --- 6. Xử lý kết quả từ chain ---
        assistant_response_content = ""
        sources = None
        
        if isinstance(chain_result, dict) and "answer" in chain_result:
            assistant_response_content = str(chain_result["answer"])
            sources = chain_result.get("context")  # Lấy sources nếu có
        elif isinstance(chain_result, str):
            assistant_response_content = chain_result
        else:
            logger.error(f"QA Chain result không hợp lệ: {chain_result}")
            assistant_response_content = "Xin lỗi, tôi không thể xử lý yêu cầu này vào lúc này."

        if not assistant_response_content.strip():
             assistant_response_content = "Tôi không tìm thấy câu trả lời phù hợp."

        # --- 7. Format sources ---
        formatted_sources = None
        if sources:
            formatted_sources = []
            for doc in sources:
                formatted_sources.append(SourceDocument(
                    source=doc.metadata.get("source", "[No source]"),
                    page_content_preview=doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else "")
                ))

    except Exception as chain_error:
        logger.error(f"Lỗi QA Chain (chat_id: {chat_id}): {chain_error}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý từ QA chain: {str(chain_error)[:100]}")

    end_time = time.time()
    logger.info(f"Trả lời cho chat {chat_id}: {assistant_response_content[:100]}...")
    
    return AnswerResponse(
        answer=assistant_response_content,
        sources=formatted_sources,
        processing_time=round(end_time - start_time, 2)
    )

async def stream_chat_generator(
    app_state,
    chat_id: str,
    question_content: str,
    chat_history: list = []
):
    """
    Refactored streaming function - removed auth/Redis/MongoDB dependencies
    Generator function to stream chat responses.
    Yields data in Server-Sent Events (SSE) format.
    """
    start_time_total = time.time()
    current_utc_time = datetime.now(timezone.utc)
    full_answer_for_saving = ""  # Để lưu toàn bộ câu trả lời

    try:
        # --- 1. Removed Redis authentication checks ---
        # No authentication or session validation needed

        # --- 2. Tiền xử lý câu hỏi ---
        cleaned_question = minimal_preprocess_for_llm(question_content)

        initial_processing_done_time = time.time()
        logger.info(f"Stream: Initial processing for {chat_id} took {initial_processing_done_time - start_time_total:.2f}s")

        # --- 3. Kiểm tra từ điển thuật ngữ (nếu có) ---
        if hasattr(app_state, 'dict') and app_state.dict:
            term_result = search_term_in_dictionary(cleaned_question, app_state.dict)
            if term_result:
                answer_def = term_result.get("definition", "Không thể tìm thấy định nghĩa.")
                assistant_response_time_dict = datetime.now(timezone.utc)
                full_answer_for_saving = answer_def

                # Stream toàn bộ định nghĩa như một chunk
                data_payload = {"token": answer_def, "is_final": True, "source": "dictionary"}
                yield f"data: {json.dumps(data_payload)}\n\n"
                # Event kết thúc
                yield f"event: end_stream\ndata: {{}}\n\n"

                # Removed Redis/MongoDB saving operations
                processing_time_dict = round(time.time() - start_time_total, 2)
                logger.info(f"Stream: Dictionary answer for {chat_id} sent in {processing_time_dict:.2f}s.")
                return

        if not app_state.qa_chain: # qa_chain phải hỗ trợ streaming
            logger.error("Stream: QA Chain chưa được khởi tạo hoặc không hỗ trợ streaming.")
            error_payload = {"error": "Service Unavailable: QA Chain not ready for streaming."}
            yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
            return

        # --- 4. Chuẩn bị chat history cho Langchain Chain ---
        try:
            # Sử dụng chat_history từ parameter thay vì Redis
            chat_history_string = format_chat_history_for_prompt(chat_history)
            input_data_for_chain = {
                "chat_history": chat_history_string,
                "input": cleaned_question
            }
        except Exception as e:
            logger.error(f"Stream: Lỗi khi chuẩn bị chat history (chat_id: {chat_id}): {e}", exc_info=True)
            error_payload = {"error": "Error processing chat history."}
            yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
            return

        # --- 5. Gọi QA Chain với streaming ---

        if not (hasattr(app_state.qa_chain, 'astream') or hasattr(app_state.qa_chain, 'stream')):
            logger.error(f"Stream: QA Chain (type: {type(app_state.qa_chain)}) không có phương thức astream hoặc stream.")
            error_payload = {"error": "QA Chain does not support streaming."}
            yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
            return

        chain_stream_method = app_state.qa_chain.astream if hasattr(app_state.qa_chain, 'astream') else app_state.qa_chain.stream

        logger.info(f"Stream: Invoking chain stream for {chat_id}...")
        stream_start_time = time.time()
        chunk_count = 0
        sources_streamed = False # Cờ để chỉ stream sources một lần

        async for chunk in chain_stream_method(input_data_for_chain):

            token = ""
            current_sources = None

            if isinstance(chunk, str):
                token = chunk
            elif hasattr(chunk, 'content'): # Giống AIMessageChunk
                token = chunk.content
            elif isinstance(chunk, dict):
                token = chunk.get("answer") or chunk.get("token") or chunk.get("content") or ""
                # Kiểm tra sources nếu chunk là dict và chưa stream sources
                if not sources_streamed and "source" in chunk:
                    current_sources = chunk["source"]

            if token:
                full_answer_for_saving += token
                data_payload = {"token": token, "is_final": False}
                yield f"data: {json.dumps(data_payload)}\n\n"
                chunk_count += 1

            # Stream sources nếu có và chưa được stream
            if current_sources and not sources_streamed:
                sources_list = []
                for doc in current_sources:
                    if hasattr(doc, 'metadata') and hasattr(doc, 'page_content'):
                        sources_list.append(SourceDocument(
                            source=doc.metadata.get('source', 'N/A'),
                            page_content_preview=doc.page_content[:200] + "..."
                        ).dict()) # Chuyển sang dict để JSON serialize
                if sources_list:
                    source_payload = {"sources": sources_list}
                    yield f"event: sources\ndata: {json.dumps(source_payload)}\n\n" # Event riêng cho sources
                    sources_streamed = True # Đánh dấu đã stream


        stream_end_time = time.time()
        logger.info(f"Stream: Chain streaming for {chat_id} completed in {stream_end_time - stream_start_time:.2f}s with {chunk_count} chunks.")

        # --- Gửi event kết thúc stream ---
        # Frontend có thể dùng event này để biết stream đã hoàn tất.
        # Hoặc, frontend có thể dựa vào một chunk đặc biệt như `{"is_final": true}`
        # Hoặc đơn giản là khi `EventSource.onmessage` không nhận được gì nữa sau một timeout.
        yield f"event: end_stream\ndata: {{ \"message\": \"Stream ended\" }}\n\n"


        # --- 6. Hoàn tất streaming ---
        assistant_response_time = datetime.now(timezone.utc)
        if not full_answer_for_saving.strip() and chunk_count == 0:
            full_answer_for_saving = "Tôi không tìm thấy câu trả lời phù hợp."
            # Stream câu trả lời mặc định này nếu chưa có gì
            data_payload = {"token": full_answer_for_saving, "is_final": True}
            yield f"data: {json.dumps(data_payload)}\n\n"
            yield f"event: end_stream\ndata: {{ \"message\": \"Stream ended with default message\" }}\n\n"

        logger.info(f"Stream: Full answer for {chat_id}: {full_answer_for_saving[:100]}...")
        # Removed Redis/MongoDB saving operations


    except HTTPException as e: # Bắt HTTPException đã được raise từ các hàm con
        logger.error(f"Stream: HTTPException for chat_id {chat_id}: {e.detail}", exc_info=True)
        error_payload = {"error": e.detail, "status_code": e.status_code}
        yield f"event: error_stream\ndata: {json.dumps(error_payload)}\n\n"
    except Exception as e:
        logger.error(f"Stream: Unhandled exception for chat_id {chat_id}: {e}", exc_info=True)
        error_payload = {"error": "An unexpected server error occurred during streaming."}
        yield f"event: error_stream\ndata: {json.dumps(error_payload)}\n\n"
    finally:
        # Đảm bảo generator kết thúc đúng cách.
        # EventSource trên client sẽ tự động đóng khi generator kết thúc.
        # Hoặc bạn có thể gửi một tín hiệu đóng rõ ràng nếu cần.
        # yield "event: close\ndata: Connection closed by server\n\n" # Không chuẩn SSE, nhưng một số client có thể hiểu
        logger.info(f"Stream: Generator for chat_id {chat_id} finished. Total time: {time.time() - start_time_total:.2f}s")


# Sử dụng GET cho EventSource theo chuẩn, truyền params qua query string
# EventSource chỉ hỗ trợ GET. Nếu bạn BẮT BUỘC phải dùng POST (ví dụ, câu hỏi quá dài cho URL),
# bạn sẽ cần một giải pháp phức tạp hơn, không dùng EventSource trực tiếp trên client
# mà dùng fetch API với ReadableStream và POST.


#helper

# Removed unused imports for Redis-based functions
# from typing import List, Optional,Any
# from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
# COMMENTED OUT - Original Redis-based function, replaced with simple version
# async def prepare_chat_history_optimized(
#     redis:Any,
#     chat_id: str,
#     max_messages: int = 10,
#     max_tokens: Optional[int] = None,
#     tokenizer: Optional[Any] = None
# ) -> List[BaseMessage]:
#     """Original Redis-based function - not used anymore"""
#     pass

def prepare_chat_history_simple(chat_history: list, max_messages: int = 10) -> list:
    """
    Simplified chat history preparation - no Redis dependencies
    Args:
        chat_history: List of chat messages (dicts with 'role' and 'content')
        max_messages: Maximum number of messages to keep
    Returns:
        Truncated list of chat messages
    """
    if not chat_history:
        return []
    
    # Simply return the last max_messages
    return chat_history[-max_messages:] if len(chat_history) > max_messages else chat_history

def format_chat_history_for_prompt(chat_history: list) -> str:
    """
    Chuyển đổi danh sách tin nhắn thành một chuỗi văn bản duy nhất.
    Simplified version - accepts simple list of dicts with 'role' and 'content'
    """
    if not chat_history:
        return "Không có lịch sử trò chuyện."

    formatted_history = []
    for message in chat_history:
        if isinstance(message, dict):
            role = "Người dùng" if message.get("role") == "user" else "Trợ lý"
            content = message.get("content", "")
            formatted_history.append(f"{role}: {content}")
        else:
            # Fallback for other message types
            role = "Người dùng" if hasattr(message, 'role') and message.role == "user" else "Trợ lý"
            content = getattr(message, 'content', str(message))
            formatted_history.append(f"{role}: {content}")

    return "\n".join(formatted_history)