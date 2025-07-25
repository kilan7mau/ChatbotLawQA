import re
import os
from typing import List, Dict, Optional, Tuple, Union,Any
import logging
from tqdm import tqdm
import uuid
import json
from langchain_core.documents import Document
from config import LEGAL_DOC_TYPES, MAX_CHUNK_SIZE, CHUNK_OVERLAP, model_process

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

import prompt_templete
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

logger = logging.getLogger(__name__)

#Hàm cuối cùng để chuẩn hóa metadata ngay trước khi ingest vào vector store.
def filter_and_serialize_complex_metadata(documents: List[Document]) -> List[Document]:
    """Hàm cuối cùng để chuẩn hóa metadata ngay trước khi ingest vào vector store."""
    updated_documents = []
    allowed_types = (str, bool, int, float, type(None))
    serialize_keys = ["penalties", "cross_references"]

    for doc in documents:
        filtered_metadata = {}
        for key, value in doc.metadata.items():
            if key in serialize_keys and value: # Chỉ serialize nếu value không rỗng
                try:
                    filtered_metadata[key] = json.dumps(value, ensure_ascii=False, default=str)
                except TypeError:
                    logger.warning(f"Không thể serialize key '{key}' cho doc ID {doc.id}. Chuyển thành string.")
                    filtered_metadata[key] = str(value)
            elif isinstance(value, allowed_types):
                filtered_metadata[key] = value
            elif isinstance(value, list):
                filtered_metadata[key] = json.dumps(value, ensure_ascii=False)
            else:
                filtered_metadata[key] = str(value)
        doc.metadata = filtered_metadata
        updated_documents.append(doc)
    return updated_documents

#Một text splitter đơn giản để chia nhỏ các chunk quá lớn.
class SimpleTextSplitter:
    """Một text splitter đơn giản để chia nhỏ các chunk quá lớn."""
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        if not text: return []
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks

base_text_splitter = SimpleTextSplitter(chunk_size=MAX_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

def generate_structured_id(doc_so_hieu: Optional[str], structure_path: List[str], filename: str) -> str:
    """
    CẢI TIẾN: Tạo ra một chuỗi UUID v5 nhất quán từ ID có cấu trúc.
    Thêm filename để đảm bảo tính duy nhất.
    """
    # Ưu tiên so_hieu, nhưng fallback về filename để tránh "unknown-document"
    base_id = doc_so_hieu if doc_so_hieu else filename
    safe_base_id = re.sub(r'[/\s\.]', '-', base_id) # Thay thế các ký tự không an toàn
    path_str = '_'.join(filter(None, structure_path))

    # Đảm bảo unique_string_id khác nhau cho mỗi file
    unique_string_id = f"{safe_base_id}_{path_str}"

    generated_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, unique_string_id)
    return str(generated_uuid)



def general_ocr_corrections(text: str) -> str:
    """Sửa các lỗi OCR phổ biến."""
    corrections = {
        "LuatWielnam": "LuatVietnam", "LualVietnam": "LuatVietnam",
        "aflu atvistnarn.vni": "@luatvietnam.vn", "Tien ch van bán luat": "Tiện ích văn bản luật",
        "teeeeokanlbaglueloen": "",
        "Tee===": "", "Tc=e===": "", "nem": "",
        "SN Hntlin sa:": "", "HT:": "", "Hntlin sa:": "",
        r"([a-z])([A-Z])": r"\1 \2",
        "Nghịđịnh": "Nghị định", " điểu ": " điều ", "Chưong": "Chương",
        " điềm ": " điểm ", "khoån": "khoản", "Chínhphủ": "Chính phủ",
        " điềukhỏan": " điều khoản", "LuậtTổ": "Luật Tổ", "LuậtXử": "Luật Xử",
        "LuậtTrật": "Luật Trật", " điềuchỉnh": " điều chỉnh", " cá_nhân": " cá nhân",
        " tổ_chức": " tổ chức", " hành_chính": " hành chính", " giấy_phép": " giấy phép",
        " lái_xe": " lái xe", " Giao_thông": " Giao thông",
        # 'ó': '6', 'Ò': '0', 'ọ': '0', 'l': '1', 'I': '1', 'i': '1',
        # 'Z': '2', 'z': '2', 'B': '8',
    }
    for wrong, right in corrections.items():
        if wrong.startswith(r"([a-z])([A-Z])"):
            text = re.sub(wrong, right, text)
        else:
            text = text.replace(wrong, right)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

### HÀM 1: CLEAN_DOCUMENT_TEXT (Cải tiến) ###
def clean_document_text(raw_text: str) -> str:
    """
    Làm sạch văn bản luật một cách an toàn, giữ lại cấu trúc cột ở phần đầu
    và loại bỏ nhiễu một cách có mục tiêu.
    """
    if not raw_text: return ""
    lines = raw_text.splitlines()

    noise_patterns_to_remove = re.compile(
        r"|".join([
            r"LuatVietnam(?:\.vn)?", r"Tiện ích văn bản luật", r"www\.vanbanluat\.vn",
            r"Hotline:", r"Email:", r"Cơ sở dữ liệu văn bản pháp luật",
            r"Trang \d+\s*/\s*\d+", r"^\s*[=\-_*#]+\s*$", r"^\s*\[\s*Hình\s*ảnh\s*]\s*$",
        ]), re.IGNORECASE
    )
    footer_keywords = ["Nơi nhận:", "TM. CHÍNH PHỦ", "TM. BAN BÍ THƯ", "KT. BỘ TRƯỞNG", "TL. BỘ TRƯỞNG", "CHỦ TỊCH QUỐC HỘI"]

    footer_start_index = len(lines)
    for i, line in enumerate(lines):
        line_upper = line.strip().upper()
        if any(keyword.upper() in line_upper for keyword in footer_keywords):
            if "CHỦ TỊCH" in line_upper and i < len(lines) / 2 and "NƯỚC" in line_upper: continue
            footer_start_index = i
            break

    lines_before_footer = lines[:footer_start_index]

    cleaned_lines = []
    for line in lines_before_footer:
        if noise_patterns_to_remove.search(line): continue
        stripped_line = line.strip()
        if not stripped_line: continue
        cleaned_lines.append(stripped_line)

    text = "\n".join(cleaned_lines)
    text = general_ocr_corrections(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def extract_document_metadata(raw_text: str, filename: str) -> Dict[str, Any]:
    """Extract metadata and return with English keys."""
    # Default metadata with English keys
    metadata = {
        "document_number": None,
        "document_type": None,
        "document_title": None,
        "issue_date": None,
        #"issue_year": None,
        "issuing_agency": None,
        "effective_date": None,
        "source_file": os.path.splitext(filename)[0],
        "confidential_level": "Công Khai"  # ✅ Mặc định nếu không tìm thấy
    }

    if not GOOGLE_API_KEY:
        raise Exception("GOOGLE_API_KEY is not set. Please provide your Google API key to extract document metadata.")

    try:
        llm = ChatGoogleGenerativeAI(
            model=model_process,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.0,
        )
        keyword_extraction_prompt = ChatPromptTemplate.from_template(
            prompt_templete.KEYWORD_EXTRACTION_PROMPT
        )
        chain = keyword_extraction_prompt | llm | JsonOutputParser()
        result = chain.invoke({"raw_text": raw_text})
        # Map Vietnamese keys to English
        mapping = {
            "so_hieu": "document_number",
            "loai_van_ban": "document_type",
            "ten_van_ban": "document_title",
            "ngay_ban_hanh_str": "issue_date",
            #"nam_ban_hanh": "issue_year",
            "co_quan_ban_hanh": "issuing_agency",
            "ngay_hieu_luc_str": "effective_date"
        }
        for vi_key, en_key in mapping.items():
            if vi_key in result:
                metadata[en_key] = result[vi_key]
        # if metadata["issue_year"] and isinstance(metadata["issue_year"], str):
        #     try:
        #         metadata["issue_year"] = int(metadata["issue_year"])
        #     except Exception:
        #         metadata["issue_year"] = None
        logger.info(f"[Gemini] Metadata extracted: {metadata}")
        return metadata
    except Exception as e:
        logger.error(f"[Gemini] Error extracting metadata: {e}")
        raise e

# --- Các hàm cũ đã được thay thế bằng LLM, giữ lại để tham khảo ---
# def extract_cross_references(text_chunk_content: str, current_doc_full_metadata: Dict) -> List[Dict]:
#     references = []
#     internal_ref_patterns = [
#         re.compile(r"(?:quy định tại|xem|theo|như|tại)\s+(?:(điểm\s+[a-zđ])\s+)?(?:(khoản\s+\d+)\s+)?(Điều\s+\d+[a-z]?)(?:\s+của\s+(?:Nghị định|Luật|Bộ luật|Thông tư|Pháp lệnh|Quyết định|Nghị quyết)\s+này)?", re.IGNORECASE),
#         re.compile(r"(?:quy định tại|xem|theo|như|tại)\s+(?:(khoản\s+\d+)\s+)?(Điều\s+\d+[a-z]?)(?:\s+của\s+(?:Nghị định|Luật|Bộ luật|Thông tư|Pháp lệnh|Quyết định|Nghị quyết)\s+này)?", re.IGNORECASE),
#         re.compile(r"(?:quy định tại|xem|theo|như|tại)\s+(Điều\s+\d+[a-z]?)(?:\s+của\s+(?:Nghị định|Luật|Bộ luật|Thông tư|Pháp lệnh|Quyết định|Nghị quyết)\s+này|\s+nêu trên|\s+dưới đây)?", re.IGNORECASE),
#     ]
#     # ... (phần internal_ref_patterns giữ nguyên) ...
#     for pattern in internal_ref_patterns:
#         for match in pattern.finditer(text_chunk_content):
#             original_text_internal = match.group(0) # Lấy toàn bộ chuỗi khớp
#             groups = match.groups()
#             ref_diem, ref_khoan, ref_dieu = None, None, None

#             # Logic xác định điểm, khoản, điều dựa trên số lượng group và nội dung
#             # Pattern 1: (điểm)? (khoản)? (Điều)
#             if pattern.pattern.count('(') - pattern.pattern.count('?:') == 3: # Đếm số capturing groups
#                 ref_diem_text = groups[0] if groups[0] and "điểm" in groups[0].lower() else None
#                 ref_khoan_text = groups[1] if groups[1] and "khoản" in groups[1].lower() else None
#                 ref_dieu_text = groups[2] if groups[2] and "điều" in groups[2].lower() else None

#                 # Nếu group 1 là khoản, group 2 là điều (do điểm optional)
#                 if not ref_diem_text and ref_khoan_text is None and (groups[0] and "khoản" in groups[0].lower()):
#                     ref_khoan_text = groups[0]
#                     ref_dieu_text = groups[1]
#                 # Nếu group 1 là điều (do điểm và khoản optional)
#                 elif not ref_diem_text and not ref_khoan_text and (groups[0] and "điều" in groups[0].lower()):
#                     ref_dieu_text = groups[0]


#             # Pattern 2: (khoản)? (Điều)
#             elif pattern.pattern.count('(') - pattern.pattern.count('?:') == 2:
#                 ref_khoan_text = groups[0] if groups[0] and "khoản" in groups[0].lower() else None
#                 ref_dieu_text = groups[1] if groups[1] and "điều" in groups[1].lower() else None
#                 # Nếu group 0 là điều
#                 if not ref_khoan_text and (groups[0] and "điều" in groups[0].lower()):
#                     ref_dieu_text = groups[0]

#             # Pattern 3: (Điều)
#             elif pattern.pattern.count('(') - pattern.pattern.count('?:') == 1:
#                 ref_dieu_text = groups[0] if groups[0] and "điều" in groups[0].lower() else None

#             ref_dieu = ref_dieu_text.replace("Điều ", "").strip() if ref_dieu_text else None
#             ref_khoan = ref_khoan_text.replace("khoản ", "").strip() if ref_khoan_text else None
#             ref_diem = ref_diem_text.replace("điểm ", "").strip() if ref_diem_text else None

#             references.append({
#                 "type": "internal", "original_text": original_text_internal,
#                 "target_dieu": ref_dieu,
#                 "target_khoan": ref_khoan,
#                 "target_diem": ref_diem,
#                 "target_document_id": current_doc_full_metadata.get("so_hieu"),
#                 "target_document_title": current_doc_full_metadata.get("ten_van_ban")
#             })


#     external_ref_pattern = re.compile(
#         r"(?:quy định tại|theo|tại|của|trong)\s+"
#         # Group 1: Cụm điểm/khoản/điều (optional), ví dụ "điểm a khoản 1 Điều 5 " hoặc "Điều 5 "
#         r"((?:điểm\s+[a-zđ]\s*)?(?:khoản\s+\d+\s*)?(?:Điều\s+\d+[a-z]?\s*)?)?"
#         r"(?:của\s+)?" # Non-capturing "của "
#         # Group 2: Loại VB + Tên VB, ví dụ "Luật Giao thông đường bộ" hoặc "Nghị định 100/2019/NĐ-CP"
#         r"((?:Nghị định|Luật|Bộ luật|Thông tư|Pháp lệnh|Quyết định|Nghị quyết|Hiến pháp)"
#         r"(?:\s+[\w\sÀ-Ỹà-ỹ\d()/'.,-]+?)?)"
#         # Group 3: Số hiệu (optional), ví dụ "100/2019/NĐ-CP"
#         r"(?:\s+(?:số|số hiệu)?\s*([\w\d/.-]+(?:-\d{4}-[\w\d.-]+)?))?"
#         # Group 4: Năm từ ngày ban hành (optional), ví dụ "2019"
#         r"(?:\s*ngày\s*\d{1,2}\s*(?:tháng|-|/)\s*\d{1,2}\s*(?:năm|-|/)\s*(\d{4}))?"
#         r"(?:\s*của\s*(?:Chính phủ|Quốc hội|[\w\sÀ-Ỹà-ỹ]+))?", # Non-capturing cơ quan ban hành
#         re.IGNORECASE
#     )

#     for match in external_ref_pattern.finditer(text_chunk_content):
#         # match.groups() sẽ trả về 4 phần tử tương ứng với 4 capturing groups ở trên
#         matched_groups = match.groups()
#         original_text_external = match.group(0) # Toàn bộ chuỗi khớp

#         provision_elements_str = matched_groups[0] if matched_groups[0] else ""
#         target_doc_full_name_raw = matched_groups[1].strip() if matched_groups[1] else ""
#         target_doc_number_explicit = matched_groups[2].strip() if matched_groups[2] else None
#         target_doc_year_in_ref_str = matched_groups[3].strip() if matched_groups[3] else None

#         # Phân tích provision_elements_str để lấy điểm, khoản, điều
#         target_diem = None
#         if diem_match_obj := re.search(r"điểm\s+([a-zđ])", provision_elements_str, re.IGNORECASE):
#             target_diem = diem_match_obj.group(1)

#         target_khoan = None
#         if khoan_match_obj := re.search(r"khoản\s+(\d+)", provision_elements_str, re.IGNORECASE):
#             target_khoan = khoan_match_obj.group(1)

#         target_dieu = None
#         if dieu_match_obj := re.search(r"Điều\s+(\d+[a-z]?)", provision_elements_str, re.IGNORECASE):
#             target_dieu = dieu_match_obj.group(1)

#         # Phân tích loại và tên văn bản từ target_doc_full_name_raw
#         target_doc_type = None
#         target_doc_title = target_doc_full_name_raw # Gán giá trị mặc định

#         for doc_type_keyword in LEGAL_DOC_TYPES:
#             # Sử dụng \b để khớp từ chính xác hơn và re.escape để xử lý ký tự đặc biệt nếu có
#             # Khớp ở đầu chuỗi và không phân biệt hoa thường
#             if re.match(rf"^{re.escape(doc_type_keyword)}\b", target_doc_full_name_raw, re.IGNORECASE):
#                 target_doc_type = doc_type_keyword.upper() # Chuẩn hóa về chữ hoa
#                 # Loại bỏ phần loại văn bản khỏi tên, và các khoảng trắng thừa
#                 temp_title = re.sub(rf"^{re.escape(doc_type_keyword)}\b\s*", "", target_doc_full_name_raw, count=1, flags=re.IGNORECASE)
#                 target_doc_title = temp_title.strip()
#                 break

#         final_doc_number = target_doc_number_explicit
#         final_doc_year = None
#         if target_doc_year_in_ref_str:
#             final_doc_year = int(target_doc_year_in_ref_str)

#         # Nếu không có số hiệu rõ ràng, thử trích từ tên (ví dụ: "Nghị định 100/2019/NĐ-CP")
#         if not final_doc_number and target_doc_title:
#             # Cố gắng bắt số hiệu dạng X/YYYY/ABC-XYZ hoặc X-YYYY-ABC
#             number_in_title_match = re.search(r"(\d+(?:/\d{4})?/[\w.-]+(?:-[\w\d.-]+)?|\d+-\d{4}-[\w.-]+)", target_doc_title)
#             if number_in_title_match:
#                 final_doc_number = number_in_title_match.group(1)
#                 # Cập nhật lại title nếu đã lấy số hiệu ra
#                 target_doc_title = target_doc_title.replace(final_doc_number, "").strip()
#                 target_doc_title = re.sub(r"^\s*(?:số|số hiệu)\s*$", "", target_doc_title, flags=re.IGNORECASE).strip() # Bỏ "số" thừa
#                 target_doc_title = target_doc_title.replace("của", "").strip() # Bỏ "của" thừa nếu có

#         # Nếu không có năm rõ ràng, thử trích từ số hiệu hoặc tên
#         if not final_doc_year and final_doc_number:
#             year_in_number_match = re.search(r"/(\d{4})/", final_doc_number) or \
#                                    re.search(r"-(\d{4})-", final_doc_number)
#             if year_in_number_match:
#                 final_doc_year = int(year_in_number_match.group(1))

#         if not final_doc_year and target_doc_title:
#             year_in_title_match = re.search(r"(?:năm|khóa)\s+(\d{4})", target_doc_title, re.IGNORECASE) # Thêm "khóa"
#             if year_in_title_match:
#                 final_doc_year = int(year_in_title_match.group(1))

#         # Bỏ qua nếu không có loại văn bản HOẶC (cả tên văn bản VÀ số hiệu đều không có)
#         if not target_doc_type or (not target_doc_title and not final_doc_number):
#             # logger.debug(f"Skipping external ref: {original_text_external} -> Type: {target_doc_type}, Title: {target_doc_title}, Number: {final_doc_number}")
#             continue

#         references.append({
#             "type": "external",
#             "original_text": original_text_external,
#             "target_document_type": target_doc_type,
#             "target_document_title": target_doc_title if target_doc_title else None,
#             "target_document_number": final_doc_number,
#             "target_document_year": final_doc_year, # Đã là int hoặc None
#             "target_dieu": target_dieu,
#             "target_khoan": target_khoan,
#             "target_diem": target_diem,
#             "target_document_year_hint": final_doc_year # Dùng final_doc_year đã được chuẩn hóa
#         })
#     return references


def hierarchical_split_law_document(doc_obj: Document) -> List[Document]:
    """
    Chunking theo Điều bằng Gemini nếu có API key, fallback về regex nếu không.
    """

    text = doc_obj.page_content
    source_metadata = doc_obj.metadata.copy()
    filename = source_metadata.get("source_file", source_metadata.get("source", "unknown_file"))
    doc_so_hieu = source_metadata.get("document_number", source_metadata.get("so_hieu"))
    document_title = source_metadata.get("document_title", source_metadata.get("ten_van_ban", filename))
    law_field = source_metadata.get("law_field", source_metadata.get("field"))

    final_chunks: List[Document] = []

    if not GOOGLE_API_KEY:
        raise Exception("GOOGLE_API_KEY is not set. Please provide your Google API key to extract document metadata.")
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_process,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.0,
        )
        hierarchical_split_law_document_prompt = ChatPromptTemplate.from_template(
            prompt_templete.HIERARCHICAL_SPLIT_LAW_DOCUMENT_PROMPT
        )
        chain = hierarchical_split_law_document_prompt | llm | JsonOutputParser()
        result = chain.invoke({"text": text})
        if isinstance(result, list) and result:
            for i, item in enumerate(result):
                dieu_code = item.get("dieu_code")
                dieu_title = item.get("dieu_title")
                content = item.get("content")
                if not dieu_code or not content:
                    continue
                block_meta = source_metadata.copy()
                block_meta["dieu_code"] = dieu_code
                block_meta["dieu_title"] = dieu_title
                block_meta["chunk_title"] = dieu_code + (f" - {dieu_title}" if dieu_title else "")
                block_meta["document_title"] = document_title
                block_meta["law_field"] = law_field
                block_meta["entity_type"] = infer_entity_type(content, law_field)
                block_meta["penalties"] = extract_penalties_from_text(content)
                #block_meta["cross_references"] = extract_cross_references(content, source_metadata)
                block_meta["cross_references"] = extract_legal_structures_and_entities(content, source_metadata)["cross_references"]
                context_header = f"Excerpt from: {block_meta['chunk_title']}\nIn document: {document_title}"
                final_page_content = f"Full text: {block_meta['chunk_title']}\n\nContent:\n{content}"
                chunk_id = generate_structured_id(doc_so_hieu, [dieu_code], filename)
                final_chunks.append(Document(page_content=final_page_content, metadata=block_meta, id=chunk_id))
            if final_chunks:
                logger.info(f"[Gemini] Chunked {len(final_chunks)} articles for file {filename}")
                return final_chunks
    except Exception as e:
        logger.error(f"[Gemini] Error chunking by article: {e}")
        raise e


#xem lại hàm này để cải tiến sau
def infer_field(text_content: str, doc_title: Optional[str]) -> str:
    """
    CẢI TIẾN V2: Bổ sung từ khóa ngắn gọn, thông tục để xử lý câu hỏi người dùng.
    """
    safe_text_content = str(text_content) if text_content else ""
    safe_doc_title = str(doc_title) if doc_title else ""

    if not safe_doc_title and not safe_text_content:
        return "khac"

    # Chỉ cần một đoạn ngắn để tìm kiếm, giúp tăng hiệu suất
    search_text = (safe_doc_title.lower() + " " + safe_text_content[:2500].lower()).strip()
    title_lower = safe_doc_title.lower()


    field_keywords = {
        # 1. Giao thông
        "giao_thong": [
            ("trật tự, an toàn giao thông đường bộ", 12),
            ("xử phạt vi phạm hành chính trong lĩnh vực giao thông", 11),
            ("giao thông đường bộ", 10),
            ("giao thông đường sắt", 10),
            ("giấy phép lái xe", 9), ("gplx", 9),
            ("đèn tín hiệu giao thông", 8),
            ("vượt đèn đỏ", 9),
            ("nồng độ cồn", 9),
            ("quá tốc độ", 8),
            ("đăng kiểm", 7),
            ("phạt nguội", 7),
            ("xe ô tô", 5), ("ô-tô", 5),
            ("xe máy", 5), ("xe mô tô", 5),
            ("lái xe", 4),
            ("biển báo", 4),
        ],

        # 2. Hình sự
        "hinh_su": [
            ("bộ luật hình sự", 12),
            ("truy cứu trách nhiệm hình sự", 10),
            ("tội phạm", 9),
            ("khởi tố", 8), ("tố tụng hình sự", 8),
            ("điều tra hình sự", 8),
            ("tòa án nhân dân", 7),
            ("viện kiểm sát", 7),
            ("giết người", 9),
            ("cướp giật tài sản", 9),
            ("lừa đảo chiếm đoạt tài sản", 9),
            ("ma túy", 8),
            ("cố ý gây thương tích", 8),
            ("tử hình", 6), ("tù chung thân", 6),
        ],

        # 3. Dân sự
        "dan_su": [
            ("bộ luật dân sự", 12),
            ("bồi thường thiệt hại ngoài hợp đồng", 10),
            ("giao dịch dân sự", 9),
            ("quyền sở hữu", 8),
            ("thừa kế", 9),
            ("di chúc", 9),
            ("hợp đồng dân sự", 7),
            ("tranh chấp dân sự", 6),
            ("nghĩa vụ dân sự", 6),
            ("đại diện, ủy quyền", 5),
        ],

        # 4. Hôn nhân và Gia đình
        "hon_nhan_gia_dinh": [
            ("luật hôn nhân và gia đình", 12),
            ("kết hôn", 9),
            ("ly hôn", 10),
            ("quan hệ giữa vợ và chồng", 8),
            ("tài sản chung của vợ chồng", 8), ("tài sản riêng", 8),
            ("quyền nuôi con", 9),
            ("cấp dưỡng", 8),
            ("mang thai hộ", 7),
            ("giám hộ", 6),
        ],

        # 5. Lao động
        "lao_dong": [
            ("bộ luật lao động", 12),
            ("hợp đồng lao động", 10),
            ("người lao động", 9), ("nlđ", 9),
            ("người sử dụng lao động", 9), ("nsdlđ", 9),
            ("bảo hiểm xã hội", 8), ("bhxh", 8),
            ("tiền lương", 7), ("lương tối thiểu vùng", 7),
            ("thời giờ làm việc", 6), ("thời giờ nghỉ ngơi", 6),
            ("kỷ luật lao động", 6),
            ("sa thải", 7), ("chấm dứt hợp đồng lao động", 7),
            ("an toàn, vệ sinh lao động", 6),
        ],

        # 6. Đất đai
        "dat_dai": [
            ("luật đất đai", 12),
            ("quyền sử dụng đất", 10),
            ("giấy chứng nhận quyền sử dụng đất", 9),
            ("thu hồi đất", 8),
            ("bồi thường, hỗ trợ, tái định cư", 8),
            ("quy hoạch, kế hoạch sử dụng đất", 7),
            ("sổ đỏ", 7), ("sổ hồng", 7),
            ("tranh chấp đất đai", 7),
            ("giá đất", 6),
        ],

        # 7. Doanh nghiệp & Đầu tư
        "doanh_nghiep": [
            ("luật doanh nghiệp", 12),
            ("luật đầu tư", 12),
            ("thành lập doanh nghiệp", 9),
            ("giấy chứng nhận đăng ký doanh nghiệp", 9),
            ("công ty cổ phần", 8),
            ("công ty trách nhiệm hữu hạn", 8),
            ("doanh nghiệp tư nhân", 8),
            ("vốn điều lệ", 7),
            ("cổ đông", 6), ("thành viên góp vốn", 6),
            ("phá sản", 7),
        ],

        # 8. Xây dựng & Nhà ở
        "xay_dung": [
            ("luật xây dựng", 12),
            ("luật nhà ở", 12),
            ("giấy phép xây dựng", 9),
            ("quy hoạch xây dựng", 8),
            ("chủ đầu tư", 7),
            ("dự án đầu tư xây dựng", 7),
            ("hợp đồng xây dựng", 6),
            ("chung cư", 6),
        ],

        # 9. Hành chính
        "hanh_chinh": [
            ("luật xử lý vi phạm hành chính", 10),
            ("xử phạt vi phạm hành chính", 9),
            ("khiếu nại", 8),
            ("tố cáo", 8),
            ("thủ tục hành chính", 7),
            ("công chức", 7), ("viên chức", 7),
            ("cán bộ", 6),
        ],

        # 10. Thuế & Tài chính & Ngân hàng
        "tai_chinh_thue": [
            ("luật quản lý thuế", 12),
            ("luật các tổ chức tín dụng", 12),
            ("thuế giá trị gia tăng", 9), ("thuế gtgt", 9),
            ("thuế thu nhập doanh nghiệp", 9), ("thuế tndn", 9),
            ("thuế thu nhập cá nhân", 9), ("thuế tncn", 9),
            ("ngân sách nhà nước", 8),
            ("hóa đơn điện tử", 7),
            ("kế toán, kiểm toán", 7),
            ("ngân hàng", 6),
            ("trái phiếu", 6),
        ],

        # 11. Môi trường
        "moi_truong": [
            ("luật bảo vệ môi trường", 12),
            ("đánh giá tác động môi trường", 9), ("đtm", 9),
            ("ô nhiễm môi trường", 8),
            ("chất thải", 7), ("chất thải nguy hại", 7),
            ("tài nguyên nước", 6),
            ("khí thải", 6),
        ],

        # 12. Sở hữu trí tuệ
        "so_huu_tri_tue": [
            ("luật sở hữu trí tuệ", 12),
            ("quyền tác giả", 9),
            ("bản quyền", 8),
            ("quyền liên quan", 8),
            ("sáng chế", 8),
            ("nhãn hiệu", 8),
            ("chỉ dẫn địa lý", 7),
        ],

        # 13. Giáo dục
        "giao_duc": [
            ("luật giáo dục", 12),
            ("học sinh, sinh viên", 7),
            ("cơ sở giáo dục", 7),
            ("học phí", 7),
            ("tuyển sinh", 6),
            ("giáo viên", 6),
            ("bằng cấp, chứng chỉ", 5),
        ],

        # 14. Y tế
        "y_te": [
            ("luật khám bệnh, chữa bệnh", 12),
            ("bảo hiểm y tế", 10), ("bhyt", 10),
            ("dược", 8), ("thuốc", 7),
            ("trang thiết bị y tế", 7),
            ("bệnh viện", 6),
            ("an toàn thực phẩm", 6),
        ],
    }

    # ... (phần logic tính điểm và trả về giữ nguyên) ...
    field_scores = {field: 0 for field in field_keywords.keys()}
    for field, weighted_keywords in field_keywords.items():
        score = 0
        for keyword, weight in weighted_keywords:
            if doc_title and keyword in title_lower:
                score += weight * 3
            occurrences_in_text = search_text.count(keyword)
            if occurrences_in_text > 0:
                score += weight * occurrences_in_text
        field_scores[field] = score

    positive_scores = {f: s for f, s in field_scores.items() if s > 0}
    if not positive_scores:
        return "khac"

    # In ra điểm số để debug
    logger.debug(f"Field scores for query '{text_content[:50]}...': {positive_scores}")

    best_field = max(positive_scores, key=positive_scores.get)
    return best_field


#xem lại hàm này để cải tiến sau
def infer_entity_type(query_or_text: str, field: Optional[str]) -> Optional[List[str]]:
    """
    CẢI TIẾN V2: Mở rộng từ khóa, xử lý khi không có field và luôn trả về list.
    """
    text_lower = query_or_text.lower()
    entity_definitions = {
        "giao_thong": {
            "xe_oto": {"keywords": ["ô tô", "xe hơi", "xe con", "xe ô-tô"], "priority": 10},
            "xe_may": {"keywords": ["xe máy", "mô tô", "xe gắn máy", "xe 2 bánh"], "priority": 10},
            # Thêm các từ khóa ngắn gọn hơn
            "nguoi_dieu_khien": {"keywords": ["người điều khiển", "lái xe", "tài xế"], "priority": 9},
            "phuong_tien": {"keywords": ["phương tiện", "xe cộ", "xe"], "priority": 5}, # Thêm "xe cộ"
        },
        "hinh_su": {
            "nguoi_pham_toi": {"keywords": ["tội phạm", "bị can", "bị cáo", "người phạm tội", "kẻ gian"], "priority": 10},
            "nan_nhan": {"keywords": ["nạn nhân", "người bị hại"], "priority": 9},
        },
        "lao_dong": {
            "nguoi_lao_dong": {"keywords": ["người lao động", "nhân viên", "công nhân", "nlđ"], "priority": 10},
            "nguoi_su_dung_lao_dong": {"keywords": ["người sử dụng lao động", "công ty", "doanh nghiệp", "nsdlđ"], "priority": 10},
            "hop_dong_lao_dong": {"keywords": ["hợp đồng lao động", "hđlđ"], "priority": 8},
        },
        "khac": {
            "ca_nhan": {"keywords": ["cá nhân", "người", "công dân", "một người"], "priority": 7},
            "to_chuc": {"keywords": ["tổ chức", "cơ quan", "đơn vị", "công ty"], "priority": 7},
        }
    }

    found_entities = []

    # CẢI TIẾN: Nếu có field, chỉ tìm trong field đó. Nếu không, tìm trong tất cả.
    fields_to_check = [field] if field and field in entity_definitions else list(entity_definitions.keys())

    for f in fields_to_check:
        current_field_entities = entity_definitions.get(f, {})
        sorted_entities = sorted(current_field_entities.items(), key=lambda item: item[1]["priority"], reverse=True)
        for entity_type, definition in sorted_entities:
            sorted_keywords = sorted(definition["keywords"], key=len, reverse=True)
            if any(re.search(r"\b" + re.escape(keyword) + r"\b", text_lower) for keyword in sorted_keywords):
                if entity_type not in found_entities:
                    found_entities.append(entity_type)

    if not found_entities:
        return None

    # Luôn trả về một danh sách các entity tìm được
    return found_entities

# xem lại hàm này để chunking và trích xuất bằng llm sau
def parse_law_item_line(line: str) -> Tuple[Optional[str], str, str]:
    """Phân tích cấu trúc dòng một cách mạnh mẽ và có thứ tự."""
    stripped_line = line.strip()
    patterns = [
        ("phan", r"^\s*(PHẦN\s+(?:THỨ\s+[\w\sÀ-Ỹà-ỹ]+|[IVXLCDM]+|CHUNG))\s*?$"),
        ("chuong", r"^\s*(Chương\s+[IVXLCDM\d]+)\s*?$"),
        ("dieu", r"^\s*(Điều\s+\d+[a-z]?)\.?\s*(.*)"),
    ]
    for item_type, pattern_str in patterns:
        match = re.match(pattern_str, stripped_line, re.IGNORECASE)
        if match:
            if item_type in ["phan", "chuong"]:
                return item_type, match.group(1).strip(), ""
            elif item_type == "dieu":
                return item_type, match.group(1).strip(), match.group(2).strip()

    if stripped_line.isupper() and len(stripped_line.split()) > 1 and len(stripped_line) < 150:
        return "title", "", stripped_line
    if m := re.match(r"^\s*(\d+)\.\s+(.*)", stripped_line):
        return "khoan", m.group(1), m.group(2).strip()
    if m := re.match(r"^\s*([a-zđ])\)\s+(.*)", stripped_line):
        return "diem", m.group(1), m.group(2).strip()
    return None, "", stripped_line



# các hàm đằng sau này đều cần xem lại
def _normalize_money(value_str: str) -> Optional[float]:
    if not value_str: return None
    try:
        return float(value_str.replace(".", "").replace(",", "."))
    except ValueError: return None

def _normalize_duration(value_str: str, unit_str: str) -> Optional[Dict[str, Union[int, str]]]:
    if not value_str or not unit_str: return None
    try:
        value = int(value_str)
        unit = unit_str.lower()
        if unit == "tháng": return {"value": value, "unit": "months"}
        if unit == "năm": return {"value": value, "unit": "years"}
        if unit == "ngày": return {"value": value, "unit": "days"}
        return {"value": value, "unit": unit_str}
    except ValueError: return None

def extract_penalties_from_text(text_content: str) -> List[Dict]:
    """
    Trích xuất các loại hình phạt khác nhau từ một đoạn văn bản.
    Cải tiến: Sử dụng set để tránh thêm các hình phạt trùng lặp.
    """
    if not text_content:
        return []

    penalties = []
    found_original_texts = set() # Set để theo dõi các chuỗi đã tìm thấy

    # --- 1. PHẠT TIỀN ---
    # Ưu tiên bắt khoảng (từ... đến...) trước
    fine_range_pattern = r"phạt tiền từ\s*([\d\.,]+)\s*đồng\s*đến\s*([\d\.,]+)\s*đồng"
    for m in re.finditer(fine_range_pattern, text_content, re.IGNORECASE):
        original_text = m.group(0)
        if original_text not in found_original_texts:
            penalties.append({
                "type": "fine",
                "min_amount": _normalize_money(m.group(1)),
                "max_amount": _normalize_money(m.group(2)),
                "currency": "đồng",
                "original_text": original_text
            })
            found_original_texts.add(original_text)

    # Bắt các mức phạt cố định sau
    fine_fixed_pattern = r"\bphạt tiền\s*([\d\.,]+)\s*đồng\b"
    for m in re.finditer(fine_fixed_pattern, text_content, re.IGNORECASE):
        original_text = m.group(0)
        # Kiểm tra xem nó có phải là một phần của một khoảng đã tìm thấy không
        is_part_of_range = any(original_text in found_range for found_range in found_original_texts)
        if not is_part_of_range and original_text not in found_original_texts:
            penalties.append({
                "type": "fine",
                "amount": _normalize_money(m.group(1)),
                "currency": "đồng",
                "original_text": original_text
            })
            found_original_texts.add(original_text)

    # --- 2. HÌNH PHẠT TÙ ---
    prison_range_pattern = r"phạt tù từ\s*(\d+)\s*(tháng|năm|ngày)\s*đến\s*(\d+)\s*(tháng|năm|ngày)"
    for m in re.finditer(prison_range_pattern, text_content, re.IGNORECASE):
        original_text = m.group(0)
        if original_text not in found_original_texts:
            penalties.append({
                "type": "prison",
                "min_duration": _normalize_duration(m.group(1), m.group(2)),
                "max_duration": _normalize_duration(m.group(3), m.group(4)),
                "original_text": original_text
            })
            found_original_texts.add(original_text)

    prison_fixed_pattern = r"\bphạt tù\s*(\d+)\s*(tháng|năm|ngày)\b"
    for m in re.finditer(prison_fixed_pattern, text_content, re.IGNORECASE):
        original_text = m.group(0)
        is_part_of_range = any(original_text in found_range for found_range in found_original_texts)
        if not is_part_of_range and original_text not in found_original_texts:
            penalties.append({
                "type": "prison",
                "duration": _normalize_duration(m.group(1), m.group(2)),
                "original_text": original_text
            })
            found_original_texts.add(original_text)

    # Tù chung thân và tử hình
    special_prison_patterns = {
        "life_imprisonment": r"phạt tù chung thân",
        "death_penalty": r"phạt tử hình"
    }
    for p_type, pattern in special_prison_patterns.items():
        if match := re.search(pattern, text_content, re.IGNORECASE):
            original_text = match.group(0)
            if original_text not in found_original_texts:
                penalties.append({"type": "prison", "duration_type": p_type, "original_text": original_text})
                found_original_texts.add(original_text)

    # --- 3. TƯỚC QUYỀN SỬ DỤNG GIẤY PHÉP ---
    license_revocation_pattern = r"tước quyền sử dụng giấy phép lái xe\s*(?:từ|từ\s*thời\s*hạn)?\s*(\d+)\s*(tháng)\s*đến\s*(\d+)\s*(tháng)"
    for m in re.finditer(license_revocation_pattern, text_content, re.IGNORECASE):
        original_text = m.group(0)
        if original_text not in found_original_texts:
            penalties.append({
                "type": "license_revocation",
                "min_duration": _normalize_duration(m.group(1), m.group(2)),
                "max_duration": _normalize_duration(m.group(3), m.group(4)),
                "original_text": original_text
            })
            found_original_texts.add(original_text)

    # --- 4. CÁC HÌNH PHẠT KHÁC (Cảnh cáo, tịch thu, v.v.) ---
    other_patterns = {
        "warning": r"\bphạt cảnh cáo\b",
        "confiscation_object_vehicle": r"tịch thu tang vật,? phương tiện vi phạm hành chính",
        "deportation": r"\btrục xuất\b"
    }
    for p_type, pattern in other_patterns.items():
        if match := re.search(pattern, text_content, re.IGNORECASE):
            original_text = match.group(0)
            if original_text not in found_original_texts:
                penalties.append({"type": p_type, "original_text": original_text})
                found_original_texts.add(original_text)

    return penalties


def load_process_and_split_documents(folder_path: str) -> List[Document]:
    all_final_chunks = []
    if not os.path.isdir(folder_path):
        logger.error(f"Folder '{folder_path}' does not exist.")
        return all_final_chunks

    txt_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.txt')]
    if not txt_files:
        logger.warning(f"No .txt files found in '{folder_path}'.")
        return all_final_chunks

    logger.info(f"Found {len(txt_files)} .txt files. Processing...")
    for filename in tqdm(txt_files, desc="Processing files"):
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()

            if not raw_content.strip():
                logger.warning(f"File '{filename}' is empty or contains only whitespace.")
                continue

            # Trích xuất metadata gốc từ raw_content trước khi làm sạch quá nhiều
            doc_metadata_original = extract_document_metadata(raw_content, filename)
            doc_metadata_original["source_file"] = os.path.splitext(filename)[0]

            # Làm sạch nội dung để xử lý (có thể giữ lại phần đầu nếu clean_document_text được điều chỉnh)
            cleaned_content_for_processing = clean_document_text(raw_content)
            if not cleaned_content_for_processing.strip():
                logger.warning(f"File '{filename}' is empty after cleaning for processing.")
                continue

            # Suy luận lĩnh vực và các thông tin khác
            doc_metadata_original["law_field"] = infer_field(cleaned_content_for_processing, doc_metadata_original.get("document_title"))
            doc_metadata_original["entity_type"] = infer_entity_type(cleaned_content_for_processing, doc_metadata_original.get("field"))
            # Penalty sẽ được trích xuất cho từng chunk

            # Tạo đối tượng Document lớn ban đầu để truyền vào hàm chia chunk
            # Nội dung là cleaned_content, metadata là doc_metadata_original
            # Tham số id của Document này không quá quan trọng vì nó sẽ được chia nhỏ
            doc_to_split = Document(page_content=cleaned_content_for_processing, metadata=doc_metadata_original)

            chunks_from_file = hierarchical_split_law_document(doc_to_split)
            all_final_chunks.extend(chunks_from_file)

        except Exception as e:
            logger.error(f"Error processing file '{filename}': {e}", exc_info=True)

    logger.info(f"Processed {len(txt_files)} files, generated {len(all_final_chunks)} final chunks.")
    # Log kiểm tra cuối cùng trước khi trả về
    for i, chk in enumerate(all_final_chunks[:3]): # Log 3 chunk đầu tiên
        logger.debug(f"Final Chunk {i} ID: {chk.id if hasattr(chk, 'id') else 'NO ID ATTR'}, Metadata: {chk.metadata}")
        if not hasattr(chk, 'id') or not chk.id:
             logger.error(f"!!! FINAL CHECK: Chunk {i} from {chk.metadata.get('source')} is missing valid ID attribute before returning from load_process_and_split_documents.")

    return all_final_chunks


def process_single_file(file_path: str) -> List[Document]:
    """
    PHIÊN BẢN CUỐI CÙNG: Pipeline xử lý hoàn chỉnh cho một file duy nhất,
    với thứ tự xử lý được tối ưu hóa.
    """
    filename = os.path.basename(file_path)
    logger.info(f"--- Starting Full Processing Pipeline for: {filename} ---")

    try:
        # --- BƯỚC 1: ĐỌC FILE ---
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()
        if not raw_content.strip():
            logger.warning(f"File '{filename}' is empty. Skipping.")
            return []

        # --- BƯỚC 2: LÀM SẠCH VĂN BẢN ---
        # Việc làm sạch trước giúp các bước sau hoạt động chính xác hơn
        cleaned_content = clean_document_text(raw_content)
        if not cleaned_content.strip():
            logger.warning(f"File '{filename}' is empty after cleaning. Skipping.")
            return []

        logger.debug(f"File '{filename}' cleaned successfully.")

        # --- BƯỚC 3: TRÍCH XUẤT METADATA CẤP VĂN BẢN ---
        # Trích xuất từ nội dung đã được làm sạch
        doc_metadata = extract_document_metadata(cleaned_content, filename)
        doc_metadata["source"] = filename

        logger.debug(f"Extracted document metadata for '{filename}': so_hieu={doc_metadata.get('so_hieu')}, loai_van_ban={doc_metadata.get('loai_van_ban')}")
        

        # --- BƯỚC 4: SUY LUẬN LĨNH VỰC ---
        # Dựa trên nội dung sạch và tiêu đề đã trích xuất
        doc_metadata["field"] = infer_field(cleaned_content, doc_metadata.get("ten_van_ban"))

        logger.debug(f"Inferred field for '{filename}': {doc_metadata['field']}")

        # --- BƯỚC 5: TẠO ĐỐI TƯỢNG DOCUMENT VÀ CHIA CHUNK ---
        doc_to_split = Document(page_content=cleaned_content, metadata=doc_metadata)

        # Gọi hàm chia chunk phiên bản cuối cùng
        chunks_from_file = hierarchical_split_law_document(doc_to_split)

        if not chunks_from_file:
            logger.warning(f"File '{filename}' did not yield any chunks after processing.")
        else:
            logger.info(f"✅ Successfully processed '{filename}', generated {len(chunks_from_file)} chunks.")

        return chunks_from_file

    except Exception as e:
        logger.error(f"❌ A critical error occurred while processing file '{filename}': {e}", exc_info=True)
        return []

# def extract_entities_with_llm(text: str, field: Optional[str] = None) -> list:
#     """
#     Sử dụng Gemini LLM để trích xuất entity_type từ văn bản thay cho regex.
#     """
#     from langchain_google_genai import ChatGoogleGenerativeAI
#     from langchain_core.prompts import ChatPromptTemplate
#     from langchain_core.output_parsers import JsonOutputParser
#     import prompt_templete
#     import os
#     GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.5-flash-preview-05-20",
#         google_api_key=GOOGLE_API_KEY,
#         temperature=0.0,
#     )
#     # Tạo prompt cho Gemini
#     prompt = ChatPromptTemplate.from_template(prompt_templete.KEYWORD_EXTRACTION_PROMPT)
#     chain = prompt | llm | JsonOutputParser()
#     # Gọi LLM để lấy entity_type
#     try:
#         result = chain.invoke({"question": text})
#         # result là list các cụm từ khóa, bạn có thể map sang entity_type nếu cần
#         return result if isinstance(result, list) else []
#     except Exception as e:
#         logger.error(f"LLM entity extraction failed: {e}")
#         return []


def extract_legal_structures_and_entities(text: str, current_doc_full_metadata: dict = None) -> dict:
    """
    Dùng LLM để trích xuất đồng thời:
    - cross_references: Danh sách các tham chiếu pháp lý (nội bộ và ngoài văn bản)
    - penalties: Danh sách các hình phạt (phạt tiền, phạt tù, tước giấy phép, ...)
    - entity_types: Danh sách các đối tượng áp dụng (cá nhân, tổ chức, phương tiện, ...)
    - law_item_line: Nếu đoạn này là dòng tiêu đề Điều, Khoản, Điểm, Chương, Phần, ... thì trả về object mô tả loại và nội dung; nếu không thì trả về null
    """
    if not GOOGLE_API_KEY:
        raise Exception("GOOGLE_API_KEY is not set. Please provide your Google API key to extract legal structures and entities.")

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser

    # Prompt tổng hợp
    prompt_template = """
    Bạn là AI chuyên phân tích văn bản pháp luật Việt Nam. Hãy đọc đoạn văn bản sau và trả về một object JSON với các trường:
    - cross_references: Danh sách các tham chiếu pháp lý (nội bộ và ngoài văn bản, nếu có, mỗi tham chiếu là một object).
    - penalties: Danh sách các hình phạt hoặc quyền lợi (phạt tiền, phạt tù, tước giấy phép, phụ cấp, ...), mỗi hình phạt là một object.
    - entity_types: Danh sách các đối tượng áp dụng (cá nhân, tổ chức, phương tiện, chức danh, ...), mỗi entity là một chuỗi.
    - law_item_line: Nếu đoạn này là dòng tiêu đề Điều, Khoản, Điểm, Chương, Phần, ... thì trả về object dạng {{"type": ..., "code": ..., "title": ...}}; nếu không thì trả về null.
    Nếu có metadata của văn bản, bạn có thể dùng để xác định tham chiếu nội bộ.
    Đoạn văn bản:
    ---
    {text}
    ---
    Metadata (nếu có):
    {metadata}
    Chỉ trả về một object JSON hợp lệ, không giải thích gì thêm.
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatGoogleGenerativeAI(
        model=model_process,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.0,
    )
    chain = prompt | llm | JsonOutputParser()
    metadata = current_doc_full_metadata if current_doc_full_metadata else {}
    result = chain.invoke({"text": text, "metadata": metadata})
    return result
