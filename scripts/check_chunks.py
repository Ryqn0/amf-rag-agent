from amf_rag_agent.config import setup_logging
setup_logging()
from amf_rag_agent.retrieval.store import search
from amf_rag_agent.retrieval.embedder import get_embeddings

embedding = get_embeddings(['facteurs de risque climatique'])[0]
results = search(embedding, k=5)
for r in results:
    print(f"Length: {len(r['text'])} chars")
    print(r['text'])
    print()