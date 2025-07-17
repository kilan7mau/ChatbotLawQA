from sentence_transformers import SentenceTransformer

try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    print("✅ Import và tải model thành công!")
except Exception as e:
    print(f"❌ Lỗi: {e}")
#đợi tý máy nó load đã