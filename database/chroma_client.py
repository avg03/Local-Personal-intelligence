import chromadb

client = chromadb.PersistentClient(
    path="./chroma"
)

collection = client.get_or_create_collection(
    name="student_memory"
)