import os
from pymongo import MongoClient
import faiss
import numpy as np
import openai
import argparse
from dotenv import load_dotenv

load_dotenv()

# MongoDB setup
DATABASE_NAME = os.getenv("db")
COLLECTION_NAME = os.getenv("archive")
mongouri = os.getenv("client")
client = MongoClient(mongouri)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# OpenAI setup
openai.api_key = os.getenv("OPENAI_API_KEY")

# FAISS setup
DIMENSIONS = 1536  # OpenAI embedding dimensions
faiss_index = faiss.IndexFlatIP(DIMENSIONS)
id_map = {}  # Maps FAISS index to MongoDB _id

def normalize_vector(vec):
    """Normalize vector to unit length for cosine similarity."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def get_embedding(text):
    """Generate OpenAI embedding for text."""
    if not text:
        return None
    
    try:
        text = text.strip()
        if len(text) > 8191:
            text = text[:8191]
            
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        
        embedding = np.array(response.data[0].embedding, dtype="float32")
        return normalize_vector(embedding)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def rebuild_faiss_index():
    """Rebuild FAISS index from MongoDB data."""
    global faiss_index, id_map
    
    print("Rebuilding FAISS index...")
    
    # Create new index
    faiss_index = faiss.IndexFlatIP(DIMENSIONS)
    id_map = {}
    
    # Fetch all documents with embeddings
    docs = collection.find({"embedding": {"$exists": True}})
    count = 0
    
    for doc in docs:
        try:
            embedding = np.array(doc["embedding"], dtype="float32")
            embedding = normalize_vector(embedding)
            
            # Add to FAISS
            faiss_index.add(np.array([embedding], dtype="float32"))
            id_map[faiss_index.ntotal - 1] = doc["_id"]
            count += 1
        except Exception as e:
            print(f"Error processing document {doc.get('_id')}: {e}")
            continue
    
    print(f"✅ FAISS index rebuilt with {count} vectors")

def search_documents(query, top_k=5):
    """Search for similar documents using cosine similarity."""
    print(f"Processing search query: {query[:100]}...")
    
    # Generate query embedding
    query_vector = get_embedding(query)
    if query_vector is None:
        print("Failed to generate embedding for search query")
        return []
    
    # Check if FAISS index is empty
    if faiss_index.ntotal == 0:
        print("No documents in the index. Try rebuilding the index first.")
        return []
    
    try:
        # Reshape query vector
        query_vector = query_vector.reshape(1, -1)
        
        # Perform search
        print(f"Searching through {faiss_index.ntotal} documents...")
        scores, indices = faiss_index.search(query_vector, min(top_k, faiss_index.ntotal))
        
        # Process results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx in id_map:
                doc_id = id_map[idx]
                try:
                    doc = collection.find_one({"_id": doc_id})
                    if doc:
                        results.append({
                            "file_name": doc.get("file_name", "Unnamed"),
                            "category": doc.get("category", "Uncategorized"),
                            "extracted_text": doc.get("extracted_text", "No preview available"),
                            "cropped_image_path": doc.get("cropped_image_path", "No link available"),
                            "similarity_score": float(score),
                            "uploaded_by": doc.get("uploaded_by", "Unknown"),
                            "processing_status": doc.get("processing_status", "unknown")
                        })
                except Exception as e:
                    print(f"Error retrieving document {doc_id}: {e}")
                    continue
        
        # Sort by similarity score
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results
    
    except Exception as e:
        print(f"Error during search: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Vector Search Utility for Document Database")
    parser.add_argument('--search', type=str, help="Search query")
    parser.add_argument('--rebuild', action='store_true', help="Rebuild FAISS index")
    parser.add_argument('--results', type=int, default=5, help="Number of results to return")
    
    args = parser.parse_args()

    if args.rebuild:
        rebuild_faiss_index()
        return

    if not args.search:
        print("Please provide a search query using --search")
        return

    # Initialize index if not already done
    if faiss_index.ntotal == 0:
        print("Initializing search index...")
        rebuild_faiss_index()

    # Perform search
    results = search_documents(args.search, args.results)
    
    if not results:
        print("\nNo matching documents found.")
        return
    
    print("\n🔍 Search Results:")
    for i, res in enumerate(results, 1):
        print(f"\n{i}. {res['file_name']} ({res['category']})")
        print(f"   Uploaded by: {res['uploaded_by']}")
        print(f"   Status: {res['processing_status']}")
        print(f"   Similarity: {res['similarity_score']:.3f}")
        print(f"   Preview: {res['extracted_text']}")
        print(f"   Link: {res['cropped_image_path']}")

if __name__ == "__main__":
    main()