# doc_to_docx.py
import win32com.client
import os
import logging

logger = logging.getLogger(__name__)

def convert_doc_to_docx(doc_path: str, output_dir: str) -> str:
    """Chuyển file .doc sang .docx bằng Microsoft Word."""
    try:
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        doc = word.Documents.Open(os.path.abspath(doc_path))
        docx_path = os.path.join(output_dir, os.path.splitext(os.path.basename(doc_path))[0] + '.docx')
        doc.SaveAs2(os.path.abspath(docx_path), FileFormat=16)  # 16 = wdFormatXMLDocument
        doc.Close(False)
        word.Quit()
        logger.info(f"Đã chuyển đổi: {doc_path} -> {docx_path}")
        return docx_path
    except Exception as e:
        logger.error(f"Lỗi khi chuyển đổi {doc_path} sang .docx: {e}")
        raise
