import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.getcwd()))
print(os.path.abspath(os.getcwd()))
import config


# =========================
# SETTINGS
# =========================
DB_PATH = config.CHROMA_DIR  # folder where your ChromaDB persists
N_QUERY_RESULTS = 3  # number of results to show in sample query

# =========================
# CONNECT TO CHROMA
# =========================
client = PersistentClient(
    path=DB_PATH,                  # folder where data is stored
    settings=Settings(),        # optional
)

print("✅ Connected to ChromaDB at:", DB_PATH)

# =========================
# LIST COLLECTIONS
# =========================
collections = client.list_collections()
if not collections:
    print("No collections found in DB.")
    exit()

print("\nCollections found:")
for c in collections:
    print("-", c.name)

# =========================
# EXPLORE EACH COLLECTION
# =========================
for c in collections:
    collection = client.get_collection(c.name)
    print(f"\n=== Collection: {collection.name} ===")
    print("Number of items:", collection.count())
    print("Metadata:", collection.metadata)

    # Get documents, IDs, metadata, embeddings
    docs = collection.get(include=["documents", "metadatas", "embeddings"])
    print("IDs:", docs["ids"])
    print("Documents:", docs["documents"])
    print("Metadatas:", docs["metadatas"])
    print("Embeddings shape:", len(docs["embeddings"]), "x", len(docs["embeddings"][0]))

    # Show summary table
    table = pd.DataFrame({
    "id": docs["ids"],                     # ids list
    "document": docs["documents"],         # documents list
    "metadata": docs["metadatas"],         # metadatas list of dicts
    "embedding_dim": [len(e) for e in docs["embeddings"]],  # embeddings are lists
    })
    print("\nDocuments in collection:")
    print(table.head())

# =========================
# SAMPLE QUERY
# =========================
query_text = input("\nEnter a query to test retrieval: ")

for c in collections:
    collection = client.get_collection(c.name)
    results = collection.query(
        query_texts=[query_text],
        n_results=N_QUERY_RESULTS,
        include=["documents", "distances", "metadatas"]
    )

    print(f"\n--- Query results for collection '{collection.name}' ---")
    for doc, dist, meta in zip(results["documents"][0], results["distances"][0], results["metadatas"][0]):
        print(f"Distance: {dist:.4f}\nDocument: {doc}\nMetadata: {meta}\n{'-'*40}")
