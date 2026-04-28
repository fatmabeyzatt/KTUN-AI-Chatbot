import os
import sys

# SQLite fix
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from embedding_config import create_embedding_model
from langchain_chroma import Chroma
from chromadb.config import Settings

# Embedding Model
embedding_model = create_embedding_model()

# DB Connection
db_path = "./chroma_db"
print(f"DEBUG: Vectorstore path: {os.path.abspath(db_path)}")

if not os.path.exists(db_path):
    print("ERROR: chroma_db directory not found!")
    sys.exit(1)

vectorstore = Chroma(
    persist_directory=db_path,
    embedding_function=embedding_model,
    collection_name="ktun_rag"
)

query = "Ayrık Matematik dersinin bilgisayar mühendisliğindeki hocasını ve ders kodunu söyle"
print(f"\nScanning for query: '{query}'")

print("-" * 30)
print("1. SIMILARITY SEARCH RESULTS (RAW)")
print("-" * 30)

try:
    # Doğrudan query yapalım
    docs = vectorstore.similarity_search(query, k=5)

    if not docs:
        print("No documents found!")
    else:
        for i, doc in enumerate(docs):
            print(f"Document {i+1}:")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Content Preview: {doc.page_content[:300]}...")
            print("-" * 20)
            
except Exception as e:
    print(f"Error during search: {e}")
