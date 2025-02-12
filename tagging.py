import os
from pymongo import MongoClient
import faiss
import numpy as np
import openai
from google.cloud import storage
import fitz 
from docx import Document
from bson import ObjectId
import argparse
import json

# Environment Variables
DATABASE_NAME = os.getenv("db")
COLLECTION = os.getenv("archive")
mongouri = os.getenv("client")
openai.api_key = os.getenv("OPENAI_API_KEY")

# MongoDB setup
client = MongoClient(mongouri)
db = client[DATABASE_NAME]
collection = db[COLLECTION]

# Google Cloud Storage Credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r""

# Initialize FAISS index with cosine similarity
DIMENSIONS = 1536  # OpenAI Embedding size
faiss_index = faiss.IndexFlatIP(DIMENSIONS) 
id_map = {}  


def load_categories():
    """Load categories from a JSON file."""
    with open('categories.json', 'r') as file:
        data = json.load(file)
        return data['categories']

categories = load_categories()


def normalize_vector(vec):
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(vec)
    if norm == 0: 
        return vec
    return vec / norm 

def get_embedding(text):
    """Generate OpenAI embedding for given text."""
    if not text:
        print("Error: Empty text provided for embedding")
        return None
        
    try:
        # Clean and prepare the text
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

def classify_document(text):
    """Categorize document using AI and predefined categories."""
    try:
        category_list = ", ".join(categories)  
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a document classifier. only state the category"},
                {"role": "user", "content": f"Classify this document: {text[:300]}. Categories: {category_list}."}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error classifying document: {e}")
        return "Unclassified"

def upload_to_gcs(bucket_name, file_path, destination_blob_name):
    """Upload file to Google Cloud Storage and return GCS URL."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        
        blob.upload_from_filename(file_path)
        return f"gs://{bucket_name}/{destination_blob_name}"
    except Exception as e:
        print(f"Error uploading to GCS: {e}")
        return None

def process_document(file_path, title):
    """Extract text, classify, generate embedding, store in MongoDB & FAISS."""
    # Upload to GCS
    gcs_link = upload_to_gcs("red_ss_cloud_test_bucket", file_path, f"documents/{title}")
    if gcs_link is None:
        print(f"Failed to upload {title} to GCS")
        return
    
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text("text")
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return
    
    # Generate and store embedding
    embedding = get_embedding(text)
    if embedding is None:
        print(f"Failed to generate embedding for {title}")
        return
    
    # Categorize and store in MongoDB
    category = classify_document(text)
    doc_data = {
        "title": title,
        "category": category,
        "snippet": text[:300],
        "storage_link": gcs_link,
        "embedding": embedding.tolist()  # Store embedding in MongoDB for backup
    }
    doc_id = collection.insert_one(doc_data).inserted_id

    # Add to FAISS index
    faiss_index.add(np.array([embedding], dtype="float32"))
    id_map[faiss_index.ntotal - 1] = doc_id

    print(f"✅ {title} processed successfully (FAISS index: {faiss_index.ntotal - 1})")

def search_documents(query, top_k=5):
    """Search for similar documents using cosine similarity."""
    print(f"Processing search query: {query[:100]}...")  # Debug logging
    
    # Create query embedding
    query_vector = get_embedding(query)
    if query_vector is None:
        print("Failed to generate embedding for search query")
        return []
    
    if faiss_index.ntotal == 0:
        print("No documents in the index. Try rebuilding the index first.")
        return []
    
    try:
        # Reshape and normalize query vector
        query_vector = query_vector.reshape(1, -1)
        
        # Perform search
        print(f"Searching through {faiss_index.ntotal} documents...")  # Debug logging
        scores, indices = faiss_index.search(query_vector, min(top_k, faiss_index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx in id_map:
                doc_id = id_map[idx]
                try:
                    doc = collection.find_one({"_id": doc_id})
                    if doc:
                        results.append({
                            "title": doc.get("title", "Untitled"),
                            "category": doc.get("category", "Uncategorized"),
                            "snippet": doc.get("snippet", "No preview available"),
                            "link": doc.get("storage_link", "No link available"),
                            "similarity_score": float(score)
                        })
                except Exception as e:
                    print(f"Error retrieving document {doc_id}: {e}")
                    continue
        
        # this sorts the results by how similer they are to the query
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results
    
    except Exception as e:
        print(f"Error during search: {e}")
        return []

def rebuild_faiss_index():
    """Rebuild FAISS index from MongoDB data."""
    global faiss_index, id_map
    
    # Make new index
    faiss_index = faiss.IndexFlatIP(DIMENSIONS)
    id_map = {}
    
    # Get all documents with embeddings
    docs = collection.find({"embedding": {"$exists": True}})
    
    for doc in docs:
        embedding = np.array(doc["embedding"], dtype="float32")
        embedding = normalize_vector(embedding)  # Ensure normalization
        
        # Add to FAISS
        faiss_index.add(np.array([embedding], dtype="float32"))
        id_map[faiss_index.ntotal - 1] = doc["_id"]
    
    print(f"✅ FAISS index rebuilt with {faiss_index.ntotal} vectors")

def main():
    parser = argparse.ArgumentParser(description="Document Classification and Search")
    parser.add_argument('--file', type=str, help="Path to the document for classification")
    parser.add_argument('--search', type=str, help="Query to search for documents")
    parser.add_argument('--rebuild-index', action='store_true', help="Rebuild FAISS index from MongoDB")
    parser.add_argument('--top-k', type=int, default=5, help="Number of results to return")
    
    args = parser.parse_args()

    if args.rebuild_index:
        rebuild_faiss_index()
    elif args.file:
        process_document(args.file, os.path.basename(args.file))
    elif args.search:
        print("\nSearching for documents...")
        results = search_documents(args.search, args.top_k)
        
        if not results:
            print("No matching documents found.")
        else:
            print("\n🔍 Search Results:")
            for i, res in enumerate(results, 1):
                print(f"\n{i}. {res['title']} ({res['category']})")
                print(f"   Similarity: {res['similarity_score']:.3f}")
                print(f"   Snippet: {res['snippet']}...")
                print(f"   Link: {res['link']}")
    else:
        print("Please provide either --file, --search, or --rebuild-index argument.")

# FAISS index on startup
print("Initializing search index...")
rebuild_faiss_index()

if __name__ == "__main__":
    main()