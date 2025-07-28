import logging
from functools import lru_cache
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_base_cross_encoder_model():
    """
    Tải và cache model cross-encoder cơ bản.
    Model chỉ được tải một lần và tái sử dụng.
    """
    logger.info(f"🧠 Loading Cross-Encoder model '{config.RERANKER_MODEL_NAME}'...")
    try:
        model = HuggingFaceCrossEncoder(model_name=config.RERANKER_MODEL_NAME)
        logger.info("✅ Cross-Encoder model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"❌ Could not load Cross-Encoder model: {e}", exc_info=True)
        raise


def get_reranker_compressor(top_n: int = 4):
    """
    Tạo CrossEncoderReranker với top_n tùy chỉnh.
    Model được cache, chỉ tạo mới compressor với top_n khác nhau.
    """
    try:
        # Lấy model đã cache
        base_model = _get_base_cross_encoder_model()

        # Tạo compressor với top_n tùy chỉnh
        compressor = CrossEncoderReranker(model=base_model, top_n=top_n)

        logger.info(f"✅ Re-ranker compressor ready with top_n={top_n}")
        return compressor
    except Exception as e:
        logger.error(f"❌ Could not create Re-ranker compressor: {e}")
        raise