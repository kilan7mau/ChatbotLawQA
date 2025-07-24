import weaviate
from weaviate.connect import ConnectionParams
from weaviate import WeaviateClient
from weaviate.classes.init import Auth
import logging
import config
import weaviate.classes.query as wvc_query

logger = logging.getLogger(__name__)


def run_diagnostic_checks(client: WeaviateClient):
    """
    Thực hiện một loạt các kiểm tra chẩn đoán trên collection chính
    để xác minh tình trạng dữ liệu.
    """
    collection_name = config.WEAVIATE_COLLECTION_NAME
    if not client.collections.exists(collection_name):
        logger.error(f"‼️ Collection '{collection_name}' không tồn tại. Không thể chạy chẩn đoán.")
        return

    collection = client.collections.get(collection_name)
    logger.info("\n" + "="*50)
    logger.info("  RUNNING DATABASE DIAGNOSTIC CHECKS")
    logger.info("="*50)

    # --- 1. Thống kê tổng quan ---
    try:
        total = collection.aggregate.over_all(total_count=True)
        logger.info(f"[STATS] Tổng số chunk: {total.total_count}")

        fields_to_check = ["giao_thong", "hinh_su", "lao_dong", "dat_dai", "y_te", "hon_nhan_gia_dinh", "doanh_nghiep", "khac"]
        for field in fields_to_check:
            try:
                field_filter = wvc_query.Filter.by_property("field").equal(field)
                response = collection.aggregate.over_all(filters=field_filter, total_count=True)
                # Chỉ in ra nếu có kết quả
                if response.total_count > 0:
                    logger.info(f"[STATS] Lĩnh vực '{field}': {response.total_count} chunks")
            except Exception:
                pass # Bỏ qua nếu có lỗi khi kiểm tra field không tồn tại
    except Exception as e:
        logger.error(f"[STATS] Lỗi khi lấy thống kê tổng quan: {e}")

    # --- 2. Kiểm tra sự tồn tại của dữ liệu Nghị định Giao thông ---
    # Thay tên file này cho đúng với file bạn đã ingest
    target_source_file = "NĐ-CP.txt"
    try:
        source_filter = wvc_query.Filter.by_property("source").equal(target_source_file)
        response = collection.query.fetch_objects(limit=1, filters=source_filter)
        if response.objects:
            logger.info(f"✅ [DATA CHECK] Tìm thấy dữ liệu từ nguồn '{target_source_file}'.")
        else:
            logger.error(f"‼️ [DATA CHECK] KHÔNG tìm thấy dữ liệu từ nguồn '{target_source_file}'. "
                         f"Đây có thể là nguyên nhân chính gây lỗi.")
    except Exception as e:
        logger.error(f"‼️ [DATA CHECK] Lỗi khi kiểm tra source '{target_source_file}': {e}")

    # --- 3. Kiểm tra tìm kiếm từ khóa pháp lý cốt lõi ---
    target_keyword = "không chấp hành hiệu lệnh của đèn tín hiệu giao thông"
    try:
        bm25_response = collection.query.bm25(
            query=target_keyword,
            query_properties=["text"],
            limit=1
        )
        if bm25_response.objects:
            logger.info(f"✅ [KEYWORD CHECK] Tìm thấy chunk chứa từ khóa '{target_keyword[:30]}...'.")
            # Log ra chi tiết chunk đầu tiên tìm được
            first_match = bm25_response.objects[0]
            logger.info(f"    -> Chunk Source: {first_match.properties.get('source')}")
            logger.info(f"    -> Chunk Title: {first_match.properties.get('title')}")
        else:
            logger.error(f"‼️ [KEYWORD CHECK] KHÔNG tìm thấy chunk nào chứa từ khóa '{target_keyword}'. "
                         f"Vấn đề có thể nằm ở quá trình chunking hoặc cleaning.")
    except Exception as e:
         logger.error(f"‼️ [KEYWORD CHECK] Lỗi khi tìm kiếm từ khóa: {e}")

    logger.info("="*50)
    logger.info("  DIAGNOSTIC CHECKS COMPLETE")
    logger.info("="*50 + "\n")

def connect_to_weaviate(run_diagnostics: bool = True) -> WeaviateClient:
    """
    Connect to Weaviate instance.
    """

    try:

        if config.WEAVIATE_URL and config.WEAVIATE_API_KEY:
            logger.info(f"Connecting to Weaviate Cloud Services at {config.WEAVIATE_URL}")
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=config.WEAVIATE_URL,
                auth_credentials=Auth.api_key(config.WEAVIATE_API_KEY),
            )
        else:
            client = WeaviateClient(
                connection_params=ConnectionParams.from_url(
                    url="http://weaviate:8080",  # Trong Docker
                    grpc_port=50051  # Cổng gRPC mặc định của Weaviate
                )
            )
            client.connect()

        # Kiểm tra kết nối
        if not client.is_connected():
            raise Exception("Không thể kết nối tới Weaviate.")
        logger.info(f"✅ Kết nối tới Weaviate tại  thành công.")
        # Chạy kiểm tra chẩn đoán nếu được yêu cầu
        if run_diagnostics:
            run_diagnostic_checks(client)
        return client
    except Exception as e:
        logger.error(f"🔸Lỗi kết nối tới Weaviate: {e}")
        client.close()
        return None
