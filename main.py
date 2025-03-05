from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import openai
import chromadb
from chromadb.utils import embedding_functions
import nltk
from nltk.tokenize import sent_tokenize
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables securely
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Download necessary NLTK data
nltk.download("punkt")

# FastAPI App Initialization
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure correct ChromaDB version
print(f"ChromaDB Version: {chromadb.__version__}")

# Define embedding function using SentenceTransformers
EMBED_MODEL = "all-MiniLM-L6-v2"
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

# Initialize ChromaDB with embedding functions
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "pdf_text_collection"
collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=embedding_func)

# OpenAI Assistant Setup
client = openai.OpenAI()
assistant = client.beta.assistants.create(
    name="AI Psychologist",
    instructions="Provide emotional support and stress management advice.",
    tools=[],
    model="gpt-4o-mini",
)

# Pydantic model for request handling
class QueryPayload(BaseModel):
    query: str
    thread_id: str | None = None


# Function for text chunking
def agentic_chunking(text: str, max_chunk_size: int = 512):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# Function to retrieve similar documents from ChromaDB
def search_similar_text(query_text, top_k=10, max_distance=1.0):
    try:
        query_embedding = embedding_func([query_text])[0]  # Correct extraction
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "distances"]
        )

        retrieved_contexts = [
            meta.get("text", "No text available")
            for meta, distance in zip(results.get("metadatas", [[]])[0], results.get("distances", [[]])[0])
            if meta and distance < max_distance
        ]

        return "\n".join(retrieved_contexts) if retrieved_contexts else "No similar documents found."

    except Exception as e:
        return f"Error retrieving documents: {str(e)}"


@app.post("/query")
async def handle_query(payload: QueryPayload):
    try:
        query = payload.query
        current_thread_id = payload.thread_id or client.beta.threads.create().id  # Store this in DB for continuity

        # Retrieve relevant context
        retrieved_context = search_similar_text(query)

        # Create a message in the thread
        message = client.beta.threads.messages.create(
            thread_id=current_thread_id,
            role="user",
            content=f"The following context is retrieved:\n{retrieved_context}\n\n{query}\n Provide only the relevant output."
        )

        # Stream Assistant Response
        with client.beta.threads.runs.stream(
                thread_id=current_thread_id,
                assistant_id=assistant.id,
                instructions="Please address the user as Amritha. The user has a premium account."
        ) as stream:
            stream.until_done()
            final_messages = stream.get_final_messages()
            response_text = final_messages[0].content[0].text.value if final_messages else "No response received."

        return {"thread_id": current_thread_id, "response": response_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
