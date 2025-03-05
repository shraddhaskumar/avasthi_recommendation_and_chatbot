import os
import chromadb
from chromadb.utils import embedding_functions
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize

# Download necessary NLTK data
nltk.download("punkt")

# Ensure correct ChromaDB version
print(f"ChromaDB Version: {chromadb.__version__}")

# Define embedding function using SentenceTransformers
EMBED_MODEL = "all-MiniLM-L6-v2"  # Use a model optimized for semantic search
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Saves the database locally

# Define ChromaDB collection with an embedding function
collection_name = "pdf_text_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=embedding_func  # Ensure embeddings are used correctly
)


# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    with open(pdf_file, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text.strip()


# Function to split text into chunks based on token limits
def split_text(text, max_tokens=4000):
    """Splits text into smaller chunks for embedding."""
    sentences = sent_tokenize(text)
    chunks, current_chunk, current_tokens = [], [], 0

    for sentence in sentences:
        sentence_tokens = len(sentence.split())

        if current_tokens + sentence_tokens > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_tokens = [sentence], sentence_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# Function to generate embeddings using SentenceTransformer
def generate_embedding(text):
    """Generates embedding for a text chunk using SentenceTransformer."""
    try:
        embedding = embedding_func([text])  # Generate embedding using SentenceTransformer
        if embedding is not None and len(embedding) > 0:
            return embedding[0]  # Return the first (and only) embedding
        else:
            print("⚠ No embedding generated!")
            return None
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return None


# Function to check stored documents in ChromaDB
def debug_collection():
    """Prints stored document metadata and embeddings to verify insertion."""
    docs = collection.get(include=["metadatas", "embeddings"])  # Include embeddings
    stored_meta = docs.get("metadatas", [])
    stored_embeddings = docs.get("embeddings", [])

    print(f"📌 Stored Document Count: {len(stored_meta)}")

    # Show only first 2 stored metadata and embeddings for debugging
    for i, (meta, embedding) in enumerate(zip(stored_meta[:2], stored_embeddings[:2])):
        print(f"\n🗂 Metadata {i + 1}: {meta}")
        print(f"🔢 Embedding {i + 1} (First 5 values): {embedding[:5]} ... [truncated]")


# Load the PDF file
pdf_file = r"C:\Users\HP\PycharmProjects\Chatbot1\stressmanagement.pdf"

# Extract text from the PDF
pdf_text = extract_text_from_pdf(pdf_file)

# Split text into chunks
texts = split_text(pdf_text)

# Upsert into ChromaDB with embeddings
print("🔄 Generating embeddings and storing in ChromaDB...")
for i, text in enumerate(texts):
    embedding = generate_embedding(text)
    if embedding is not None:  # Ensure embedding is valid
        collection.add(
            ids=[f"pdf-text-{i}"],
            embeddings=[embedding],
            metadatas=[{"text": text}]
        )
        print(f"✅ Stored chunk {i}: {text[:50]}...")  # Print first 50 chars of each stored chunk

print(f"✅ Successfully stored {len(texts)} text chunks in ChromaDB!")

# Debug: Check if documents were stored correctly
debug_collection()


# Function to search similar text using ChromaDB
def search_similar_text(query_text, top_k=3):
    """Searches ChromaDB for relevant text using embeddings."""
    query_embedding = generate_embedding(query_text)
    if query_embedding is None:
        return "⚠ Error generating query embedding."

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "distances"]
    )

    if not results or "metadatas" not in results or not results["metadatas"][0]:
        return "❌ No similar documents found."

    print("\n🔍 Search Results:")
    for i, (meta, distance) in enumerate(zip(results["metadatas"][0], results["distances"][0])):
        if meta is None:
            print(f"❌ Result {i + 1}: Metadata is missing (None).")
            continue  # Skip None values safely

        text = meta.get("text", "❌ No Text Available")  # Retrieve text safely
        print(f"{i + 1}. {text[:100]}... (Similarity Score: {distance:.4f})\n")


# Example usage: Query ChromaDB
query = "How to manage stress?"
search_similar_text(query)
