# minimal_retrieval_demo.py

import config
from rag_components import get_huggingface_embeddings
from db.weaviateDB import connect_to_weaviate
import weaviate.classes.query as wvc_query
from utils.synonym_map import rewrite_query_with_legal_synonyms


def embed_query(embeddings_model, query: str):
    return embeddings_model.embed_query(query)

def search_weaviate(client, collection_name, query, query_vector, top_k=5):
    collection = client.collections.get(collection_name)
    # Hybrid search: both vector and keyword
    response = collection.query.hybrid(
        query=query,
        vector=query_vector,
        limit=top_k,
        alpha=0.5,  # balance between keyword and vector
        return_metadata=wvc_query.MetadataQuery(score=True)
    )
    return response.objects

if __name__ == "__main__":
    # 1. Load embedding model
    embeddings = get_huggingface_embeddings(config.EMBEDDING_MODEL_NAME, device='cpu')
    print(f"Using embedding model: {embeddings.model_name}")

    # 2. Connect to Weaviate
    client = connect_to_weaviate(run_diagnostics=False)
    if not client:
        print("Failed to connect to Weaviate.")
        exit(1)
    print(f"Connected to Weaviate at {config.WEAVIATE_URL}")
    # 3. Get user query
    user_query = input("Enter your legal question: ")
    print(f"Query: {user_query}")
    # 4. Optionally rewrite query using legal synonyms
    rewritten_query = rewrite_query_with_legal_synonyms(user_query)
    print(f"Rewritten Query: {rewritten_query}")
    # 5. Embed the query
    query_vector = embed_query(embeddings, rewritten_query)
    
    # 6. Search Weaviate
    results = search_weaviate(client, config.WEAVIATE_COLLECTION_NAME, rewritten_query, query_vector, top_k=5)

    # 7. Print results
    print("\nTop relevant chunks:")
    for i, obj in enumerate(results, 1):
        text = obj.properties.get('text', '[No text]')
        if not isinstance(text, str):
            text = str(text)
        source = obj.properties.get('source', '[No source]')
        score = obj.metadata.score if obj.metadata else None
        print(f"\nResult {i}:")
        print(f"Source: {source}")
        print(f"Score: {score}")
        print(f"Text: {text[:500]}...")  # Print first 500 chars

    client.close()


#embeddings = get_huggingface_embeddings(config.EMBEDDING_MODEL_NAME, device='cpu')