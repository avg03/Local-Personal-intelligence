import chromadb

client = chromadb.PersistentClient(
    path="./chroma"
)

collection = client.get_or_create_collection(
    name="student_memory",
    metadata={"hnsw:space": "cosine"},
)

summary_collection = client.get_or_create_collection(
    name="summary_embeddings",
    metadata={"hnsw:space": "cosine"},  # this part matters
)