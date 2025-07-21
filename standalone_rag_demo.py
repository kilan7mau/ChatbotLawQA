import os
import config
import rag_components
from db.weaviateDB import connect_to_weaviate
from utils.AdvancedLawRetriever import AdvancedLawRetriever
from services.reranker_service import get_reranker_compressor

# Device selection: try torch, else fallback to 'cpu'
try:
    import importlib.util
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is not None:
        exec('import torch\ndevice = "cuda" if torch.cuda.is_available() else "cpu"')
    else:
        device = 'cpu'
except Exception:
    device = 'cpu'

# Optionally: from dotenv import load_dotenv

def main():
    # 1. Load environment variables (if needed)
    # from dotenv import load_dotenv
    # load_dotenv()

    print("\n=== Standalone RAG Demo ===\n")

    # 1. Load embedding model
    print("Loading embedding model...")
    embeddings = rag_components.get_huggingface_embeddings(config.EMBEDDING_MODEL_NAME, device=device)
    if not embeddings:
        print("Failed to load embedding model.")
        return

    # 2. Connect to Weaviate
    print("Connecting to Weaviate...")
    weaviate_client = connect_to_weaviate(run_diagnostics=False)
    if not weaviate_client:
        print("Failed to connect to Weaviate.")
        return

    # 3. Load LLM (Google only)
    print("Loading LLM...")
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    llm = None
    if hasattr(rag_components, 'get_google_llm') and google_api_key:
        llm = rag_components.get_google_llm(google_api_key)
    if not llm:
        print("Failed to load LLM. Please set GOOGLE_API_KEY.")
        return

    # 4. Load reranker
    print("Loading reranker...")
    reranker = get_reranker_compressor()
    if not reranker:
        print("Failed to load reranker.")
        return

    # 5. Instantiate AdvancedLawRetriever
    print("Instantiating retriever...")
    retriever = AdvancedLawRetriever(
        client=weaviate_client,
        collection_name=config.WEAVIATE_COLLECTION_NAME,
        llm=llm,
        reranker=reranker,
        embeddings_model=embeddings
    )

    # 6. Build the RAG chain
    print("Building RAG chain...")
    qa_chain = rag_components.create_qa_chain(
        llm=llm,
        retriever=retriever,
        process_input_llm=llm  # Use the same LLM for preprocessing
    )
    if not qa_chain:
        print("Failed to create QA chain.")
        return

    # 7. Accept user query and run RAG
    print("\nReady! Type your legal question (or 'exit' to quit):\n")
    while True:
        user_query = input("Your question: ").strip()
        if user_query.lower() in ("exit", "quit"): break
        input_data = {"input": user_query, "chat_history": []}  # No chat history
        try:
            result = qa_chain.invoke(input_data)
            # After result = qa_chain.invoke(input_data)
            # The result may not include classification/rewritten_question, so let's run the preprocessing step manually for transparency

            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import JsonOutputParser

            # Prepare the preprocessing prompt and parser
            from prompt_templete import UNIFIED_PREPROCESSING_PROMPT
            preprocessing_prompt = ChatPromptTemplate.from_template(UNIFIED_PREPROCESSING_PROMPT)
            parser = JsonOutputParser()

            # Run preprocessing to get classification and rewritten question
            preprocessing_chain = preprocessing_prompt | llm | parser
            preprocessing_result = preprocessing_chain.invoke({"input": user_query, "chat_history": []})

            classification = preprocessing_result.get("classification", "unknown")
            rewritten_question = preprocessing_result.get("rewritten_question", user_query)

            print(f"\n[Classification]: {classification}")
            print(f"[Rewritten Question]: {rewritten_question}")

            answer = result["answer"] if isinstance(result, dict) and "answer" in result else str(result)
            print("\n=== Answer ===\n" + answer)

            sources = result.get("context") if isinstance(result, dict) else None
            if sources:
                print("\n--- Sources ---")
                for i, doc in enumerate(sources, 1):
                    src = doc.metadata.get("source", "[No source]")
                    preview = doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else "")
                    print(f"[{i}] {src}: {preview}")
                print("\n----------------\n")
        except Exception as e:
            print(f"Error during RAG pipeline: {e}")

    weaviate_client.close()
    print("Goodbye!")

if __name__ == "__main__":
    main() 