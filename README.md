<p align="center">
  <h1 align="center">⚖️ JuriBot - Trợ lý Pháp luật Việt Nam</h1>
  <p align="center">
    <em>Chatbot tư vấn pháp luật thông minh sử dụng RAG (Retrieval-Augmented Generation)</em>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.116+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Weaviate-Vector_DB-00C853?style=for-the-badge&logo=weaviate&logoColor=white" alt="Weaviate">
  <img src="https://img.shields.io/badge/Google_Gemini-LLM-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Gemini">
  <img src="https://img.shields.io/badge/Version-2.0-FF6F00?style=for-the-badge" alt="Version">
</p>

---

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Cài đặt](#-cài-đặt)
- [Cấu hình](#-cấu-hình)
- [Sử dụng](#-sử-dụng)
- [API Endpoints](#-api-endpoints)
- [RAG Pipeline](#-rag-pipeline)
- [Xử lý dữ liệu pháp luật](#-xử-lý-dữ-liệu-pháp-luật)

---

## 🎯 Giới thiệu

**JuriBot** là hệ thống chatbot tư vấn pháp luật Việt Nam, được xây dựng dựa trên kiến trúc **RAG (Retrieval-Augmented Generation)**. Hệ thống kết hợp khả năng truy xuất văn bản pháp luật từ cơ sở dữ liệu vector với mô hình ngôn ngữ lớn (LLM) để cung cấp các câu trả lời chính xác, có trích dẫn nguồn.

### Tính năng nổi bật

- 🔍 **Tìm kiếm Hybrid** — Kết hợp BM25 (keyword) và tìm kiếm vector (semantic) trên Weaviate
- 🧠 **Reranking thông minh** — Sử dụng cross-encoder để sắp xếp lại kết quả theo độ liên quan
- 📚 **Xử lý văn bản pháp luật chuyên sâu** — Tự động phân tích cấu trúc Phần, Chương, Mục, Điều, Khoản, Điểm
- 💬 **Streaming responses** — Trả lời theo thời gian thực qua Server-Sent Events (SSE)
- 🏷️ **Phân loại tự động** — Tự động phân biệt câu hỏi pháp lý và câu hỏi thông thường
- 📖 **Từ điển pháp lý** — Hệ thống synonym map và từ điển giải nghĩa thuật ngữ pháp lý
- 🔄 **Tiền xử lý truy vấn** — Chuẩn hóa chính tả, viết lại câu hỏi dựa trên lịch sử hội thoại

---

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────┐
│                        Client (Frontend)                     │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP / SSE
┌─────────────────────▼───────────────────────────────────────┐
│                    FastAPI Server (main.py)                   │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌────────────┐ │
│  │  Chat    │  │ Document │  │   User    │  │  Health    │  │
│  │  Router  │  │  Router  │  │  Router   │  │  Check     │  │
│  └────┬─────┘  └────┬─────┘  └────┬──────┘  └────────────┘ │
│       │              │             │                          │
│  ┌────▼──────────────▼─────────────▼────────────────────┐   │
│  │              Services Layer                           │   │
│  │  ┌────────────────┐  ┌──────────────────────────┐    │   │
│  │  │  Chat Service  │  │  Document Service        │    │   │
│  │  │  (RAG Query)   │  │  (Upload & Process)      │    │   │
│  │  └───────┬────────┘  └──────────────────────────┘    │   │
│  └──────────│───────────────────────────────────────────┘   │
│             │                                                │
│  ┌──────────▼───────────────────────────────────────────┐   │
│  │              RAG Pipeline (rag_components.py)          │   │
│  │                                                       │   │
│  │  ┌─────────────────┐    ┌──────────────────────┐     │   │
│  │  │  Preprocessing  │───▶│  AdvancedLawRetriever│     │   │
│  │  │  (Gemini LLM)   │    │  - Hybrid Search     │     │   │
│  │  │  - Chuẩn hóa    │    │  - Keyword Extract   │     │   │
│  │  │  - Phân loại    │    │  - Metadata Filter   │     │   │
│  │  │  - Viết lại Q   │    │  - Cross-encoder     │     │   │
│  │  └─────────────────┘    │    Reranking          │     │   │
│  │                         └───────────┬────────────┘     │   │
│  │                                     │                  │   │
│  │  ┌──────────────────────────────────▼──────────────┐  │   │
│  │  │  QA Chain (LangChain)                            │  │   │
│  │  │  - Route: legal_rag / general_chat               │  │   │
│  │  │  - Generate answer with citations                │  │   │
│  │  └──────────────────────────────────────────────────┘  │   │
│  └───────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
                      │
        ┌─────────────┼──────────────┐
        ▼             ▼              ▼
┌──────────────┐ ┌─────────┐ ┌──────────────┐
│  Weaviate    │ │ Gemini  │ │ HuggingFace  │
│  Vector DB   │ │  API    │ │  Models      │
│  (Cloud)     │ │         │ │  - Embedding │
└──────────────┘ └─────────┘ │  - Reranker  │
                              └──────────────┘
```

---

## 🛠️ Công nghệ sử dụng

| Thành phần | Công nghệ | Mô tả |
|---|---|---|
| **Backend Framework** | FastAPI | REST API với async support, auto docs |
| **LLM chính** | Google Gemini 2.0 Flash | Sinh câu trả lời từ context |
| **LLM phụ (Preprocessing)** | Google Gemini | Chuẩn hóa & phân loại câu hỏi |
| **Embedding Model** | `bkai-foundation-models/vietnamese-bi-encoder` | Embedding tiếng Việt chuyên dụng |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Sắp xếp lại kết quả tìm kiếm |
| **Vector Database** | Weaviate Cloud | Lưu trữ & tìm kiếm vector |
| **Orchestration** | LangChain | RAG chain, prompt templates |
| **Streaming** | Server-Sent Events (SSE) | Trả lời thời gian thực |

---

## 📂 Cấu trúc dự án

```
ChatbotLawQA/
├── main.py                     # Entry point - FastAPI application
├── config.py                   # Cấu hình tập trung (env, models, paths)
├── dependencies.py             # Dependency injection & khởi tạo components
├── rag_components.py           # RAG pipeline (embeddings, vectorstore, QA chain)
├── prompt_templete.py          # Prompt templates cho các tác vụ LLM
├── build_vectorstore.py        # Script xây dựng/cập nhật Vector Store
│
├── routers/                    # API Route handlers
│   ├── chat.py                 # Endpoints chat & RAG query
│   ├── documents.py            # Endpoints upload & quản lý tài liệu
│   ├── user.py                 # Endpoints quản lý người dùng
│   └── health_check.py         # Health check endpoint
│
├── services/                   # Business logic layer
│   ├── chat_service.py         # Logic xử lý chat & streaming
│   ├── document_service.py     # Logic xử lý tài liệu
│   ├── reranker_service.py     # Service reranking kết quả
│   └── user_service.py         # Logic quản lý người dùng
│
├── schemas/                    # Pydantic models
│   └── chat.py                 # AppState, QueryRequest, AnswerResponse, ...
│
├── db/                         # Database connections
│   └── weaviateDB.py           # Kết nối & chẩn đoán Weaviate
│
├── core/                       # Core utilities
│   └── logging_config.py       # Cấu hình logging
│
├── utils/                      # Utility functions
│   ├── AdvancedLawRetriever.py # Custom retriever cho văn bản pháp luật
│   ├── process_data.py         # Xử lý & phân tích cấu trúc văn bản pháp luật
│   ├── synonym_map.py          # Bản đồ từ đồng nghĩa pháp lý
│   ├── utils.py                # Hàm tiện ích chung
│   └── doc_to_docx.py          # Chuyển đổi .doc → .docx
│
├── data/                       # Thư mục dữ liệu
│   ├── core/                   # Văn bản pháp luật gốc (.txt, .docx)
│   ├── dictionary/             # Từ điển giải nghĩa thuật ngữ
│   ├── processed_files/        # File đã xử lý
│   └── processed_files_metadata/ # Metadata của file đã xử lý
│
├── requirements.txt            # Python dependencies
├── .env.example                # Template biến môi trường
└── .env                        # Biến môi trường (không commit)
```

---

## ⚙️ Cài đặt

### Yêu cầu hệ thống

- Python 3.10+
- Tài khoản Weaviate Cloud
- Google Gemini API Key

### Các bước cài đặt

**1. Clone repository:**

```bash
git clone <repository-url>
cd ChatbotLawQA
```

**2. Tạo virtual environment:**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

**3. Cài đặt dependencies:**

```bash
pip install -r requirements.txt
```

**4. Cấu hình biến môi trường:**

```bash
cp .env.example .env
```

Chỉnh sửa file `.env` với thông tin thực tế (xem phần [Cấu hình](#-cấu-hình)).

---

## 🔧 Cấu hình

Tạo file `.env` tại thư mục gốc với các biến sau:

```env
# ===== Weaviate Cloud =====
WEAVIATE_URL="https://your-cluster.weaviate.network"
WEAVIATE_API_KEY="your-weaviate-api-key"
WEAVIATE_COLLECTION_NAME="LawDocuments"

# ===== Google Gemini =====
GEMINI_API_KEY="your-gemini-api-key"

# ===== Groq (tùy chọn) =====
GROQ_API_KEY="your-groq-api-key"

# ===== LlamaCloud (tùy chọn - cho document parsing) =====
LLAMA_CLOUD_API_KEY="your-llama-cloud-api-key"

# ===== Server =====
API_HOST="0.0.0.0"
API_PORT=5000
APP_ENVIRONMENT="development"
ALLOWED_ORIGINS="http://localhost:3000"
FRONTEND_URL="http://localhost:3000"

# ===== Redis (tùy chọn) =====
REDIS_URL="redis://localhost:6379"

# ===== Authentication (tùy chọn) =====
SECRET_KEY="your-secret-key"
ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=60
SESSION_SECRET_KEY="your-session-secret"
GOOGLE_CLIENT_ID="your-google-client-id"
GOOGLE_CLIENT_SECRET="your-google-client-secret"
```

---

## 🚀 Sử dụng

### 1. Xây dựng Vector Store

Trước khi chạy API, cần nạp dữ liệu pháp luật vào Weaviate:

```bash
# Đặt file văn bản pháp luật (.txt, .docx) vào thư mục data/core/
# Sau đó chạy:
python build_vectorstore.py
```

Script sẽ tự động:
- Đọc và phân tích cấu trúc các văn bản pháp luật
- Trích xuất metadata (số hiệu, cơ quan ban hành, ngày hiệu lực, ...)
- Chia nhỏ theo đơn vị Điều/Khoản
- Tạo embedding và nạp vào Weaviate
- Hỗ trợ checkpointing (có thể tiếp tục nếu bị gián đoạn)

### 2. Khởi chạy API Server

```bash
python main.py
```

Server sẽ chạy tại `http://localhost:5000`. Truy cập API docs tại:
- **Swagger UI**: `http://localhost:5000/docs`
- **ReDoc**: `http://localhost:5000/redoc`

---

## 📡 API Endpoints

### Chat

| Method | Endpoint | Mô tả |
|---|---|---|
| `POST` | `/api/chat` | Gửi câu hỏi, nhận câu trả lời đầy đủ |
| `GET` | `/api/chat/stream` | Streaming response qua SSE |
| `POST` | `/api/chat/rag/query` | RAG query trực tiếp |
| `POST` | `/api/chat/warmup` | Khởi động trước (preload models) |

### Health Check

| Method | Endpoint | Mô tả |
|---|---|---|
| `GET` | `/api/chat/health` | Kiểm tra trạng thái các components |
| `GET` | `/api/health` | Health check tổng quát |

### Ví dụ request

**Chat thông thường:**

```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "chat_id": "session-001",
    "input": "Mức phạt vượt đèn đỏ đối với xe máy là bao nhiêu?"
  }'
```

**Streaming:**

```bash
curl "http://localhost:5000/api/chat/stream?chat_id=session-001&input=Điều+kiện+kết+hôn+theo+luật+Việt+Nam"
```

### Ví dụ response

```json
{
  "answer": "**Lĩnh vực**: Giao thông đường bộ\n**Vấn đề**: Mức xử phạt vượt đèn đỏ đối với xe máy\n**Quy định pháp luật**:\n- Theo Điều 6, Khoản 4, Điểm e Nghị định 100/2019/NĐ-CP...",
  "sources": [
    {
      "source": "Nghị định 100/2019/NĐ-CP",
      "page_content_preview": "Điều 6. Xử phạt người điều khiển xe mô tô..."
    }
  ],
  "processing_time": 3.45
}
```

---

## 🔄 RAG Pipeline

Pipeline xử lý một truy vấn đi qua các bước sau:

### Bước 1: Tiền xử lý (Unified Preprocessing)
- **Chuẩn hóa ngôn ngữ**: Sửa lỗi chính tả, chuẩn hóa thuật ngữ pháp lý
- **Viết lại câu hỏi**: Kết hợp lịch sử hội thoại để tạo câu hỏi độc lập
- **Phân loại**: Xác định `legal_rag` (pháp lý) hoặc `general_chat` (thông thường)

### Bước 2: Truy xuất (AdvancedLawRetriever)
- **Trích xuất từ khóa** bằng LLM và synonym mapping
- **Hybrid Search** trên Weaviate (BM25 + Vector, alpha=0.5)
- **Metadata Filtering**: Lọc theo loại văn bản, lĩnh vực, năm ban hành
- **Cross-encoder Reranking**: Sắp xếp lại top-K kết quả

### Bước 3: Sinh câu trả lời (QA Chain)
- **Nhánh pháp lý**: Trả lời có cấu trúc với trích dẫn Điều/Khoản cụ thể
- **Nhánh chung**: Trả lời thông thường không cần tra cứu

---

## 📜 Xử lý dữ liệu pháp luật

Hệ thống hỗ trợ xử lý các loại văn bản:

| Loại văn bản | Ví dụ |
|---|---|
| Luật, Bộ luật | Luật Giao thông đường bộ, Bộ luật Hình sự |
| Nghị định | Nghị định 100/2019/NĐ-CP |
| Thông tư | Thông tư hướng dẫn thi hành |
| Quyết định, Pháp lệnh | Quyết định của Thủ tướng |
| Nghị quyết, Chỉ thị | Nghị quyết Quốc hội |
| Hiến pháp | Hiến pháp 2013 |

### Quy trình xử lý

1. **Đọc file** — Hỗ trợ `.txt`, `.docx`, `.doc`
2. **Phân tích cấu trúc** — Sử dụng Gemini LLM để trích xuất JSON có cấu trúc
3. **Chunk theo Điều** — Mỗi Điều luật = 1 chunk với metadata đầy đủ
4. **Metadata enrichment** — Gắn thêm: lĩnh vực, cross-references, penalties, entity types
5. **Embedding & Ingest** — Vector hóa và nạp batch vào Weaviate

---

## 📝 Giấy phép

Dự án này được phát triển cho mục đích nghiên cứu và giáo dục.

---

<p align="center">
  Made with ❤️ for Vietnamese Legal AI
</p>
