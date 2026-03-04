# utils.py - Cleaned version for RAG-only API
import os
import logging
import regex as re
import json
from typing import List, Optional, Dict
from unidecode import unidecode

logger = logging.getLogger(__name__)

def load_legal_dictionary(path: str = 'legal_terms.json') -> list:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['dictionary']

def is_definition_question(query: str) -> bool:
    definition_keywords = ["là gì", "định nghĩa", "nghĩa là gì", "hiểu thế nào", "khái niệm"]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in definition_keywords)

def normalize_text_for_matching(text: str) -> str:
    """
    Chuẩn hóa text cho việc so khớp: chữ thường, loại bỏ ký tự đặc biệt (chỉ giữ chữ và số),
    loại bỏ dấu tiếng Việt, chuẩn hóa khoảng trắng.
    """
    if not text or not isinstance(text, str):
        return ""
    text_no_diacritics = unidecode(text.lower()) # Chuyển không dấu và chữ thường
    # Loại bỏ tất cả ký tự không phải là chữ cái hoặc số hoặc khoảng trắng
    text_alphanumeric = re.sub(r'[^\w\s]', '', text_no_diacritics, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', text_alphanumeric).strip()

def search_term_in_dictionary(query: str, dictionary: List[Dict]) -> Optional[Dict]:
    """
    Tìm kiếm thuật ngữ trong từ điển.
    Chỉ tìm nếu là câu hỏi định nghĩa.
    Cải thiện logic so khớp.
    """
    if not is_definition_question(query):
        logger.debug(f"'{query}' không phải câu hỏi định nghĩa, bỏ qua tìm từ điển.")
        return None

    if not dictionary:
        logger.warning("Từ điển rỗng, không thể tìm kiếm.")
        return None

    # Cố gắng trích xuất thuật ngữ chính từ câu hỏi định nghĩa
    # Ví dụ: "Khái niệm hợp đồng lao động là gì?" -> "hợp đồng lao động"
    # Đây là một regex đơn giản, có thể cần tinh chỉnh
    term_to_search_raw = query
    match = re.match(r"^(.*?)\s+(là gì|định nghĩa|nghĩa là gì|hiểu thế nào|khái niệm)\??$", query.lower().strip(), re.IGNORECASE)
    if match:
        term_to_search_raw = match.group(1).strip()
        logger.info(f"Trích xuất thuật ngữ từ câu hỏi định nghĩa: '{term_to_search_raw}'")

    query_normalized_for_match = normalize_text_for_matching(term_to_search_raw)
    if not query_normalized_for_match:
        logger.debug("Thuật ngữ tìm kiếm rỗng sau khi chuẩn hóa.")
        return None

    logger.info(f"Tìm kiếm thuật ngữ đã chuẩn hóa (không dấu): '{query_normalized_for_match}'")

    # Sắp xếp từ điển theo độ dài thuật ngữ giảm dần (để ưu tiên khớp cụm dài hơn)
    # và chuẩn hóa thuật ngữ từ điển một lần
    normalized_dictionary = []
    for entry in dictionary:
        term = entry.get("term")
        if term and isinstance(term, str):
            normalized_dictionary.append({
                "original_entry": entry,
                "normalized_term": normalize_text_for_matching(term)
            })

    # Sắp xếp theo độ dài thuật ngữ đã chuẩn hóa giảm dần
    # Điều này giúp "an toàn lao động" được khớp trước "an toàn" hoặc "lao động"
    # nếu query là "an toàn lao động là gì"
    normalized_dictionary.sort(key=lambda x: len(x["normalized_term"]), reverse=True)

    # Tìm kiếm khớp chính xác (sau khi chuẩn hóa cả query và term từ điển)
    for item in normalized_dictionary:
        if item["normalized_term"] == query_normalized_for_match:
            logger.info(f"Tìm thấy khớp chính xác (sau chuẩn hóa): '{item['original_entry']['term']}'")
            return item["original_entry"]

    # Tìm kiếm "chứa" (thuật ngữ từ điển là một phần của query đã chuẩn hóa)
    # Điều này hữu ích nếu query_normalized_for_match dài hơn thuật ngữ từ điển
    # Ví dụ: query_normalized = "dinh nghia an toan lao dong", term_normalized = "an toan lao dong"
    for item in normalized_dictionary:
        if item["normalized_term"] and item["normalized_term"] in query_normalized_for_match:
            logger.info(f"Tìm thấy khớp 'chứa' (từ điển trong query): '{item['original_entry']['term']}' (query norm: '{query_normalized_for_match}')")
            return item["original_entry"]

    logger.info(f"Không tìm thấy thuật ngữ '{query_normalized_for_match}' trong từ điển.")
    return None

def minimal_preprocess_for_llm(text: str) -> str:
    """
    Thực hiện tiền xử lý tối thiểu trước khi đưa vào LLM.
    Chỉ chuẩn hóa khoảng trắng và chuyển thành chữ thường.
    """
    if not text or not text.strip():
        # Vẫn cần kiểm tra input rỗng
        raise ValueError("Input không được rỗng")

    # 1. Chuẩn hóa khoảng trắng
    processed_text = re.sub(r'\s+', ' ', text).strip()

    # 2. Chuyển thành chữ thường để nhất quán
    processed_text = processed_text.lower()

    return processed_text

# Hash calculation functions for file processing
import hashlib
import config

def calculate_file_hash(filepath: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def check_if_hash_exists(file_hash: str) -> bool:
    if not os.path.exists(config.PROCESSED_HASH_LOG):
        return False
    try:
        with open(config.PROCESSED_HASH_LOG, "r") as f:
            processed_hashes = {line.strip() for line in f}
            return file_hash in processed_hashes
    except IOError as e:
        logger.error(f"Could not read hash log file: {e}")
        return False

def log_processed_hash(file_hash: str):
    try:
        with open(config.PROCESSED_HASH_LOG, "a") as f:
            f.write(file_hash + "\n")
    except IOError as e:
        logger.error(f"Could not write to hash log file: {e}")