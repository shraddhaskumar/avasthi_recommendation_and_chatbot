import os
import openai
import chromadb
from chromadb.utils import embedding_functions

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = "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"  # Replace with a secure method
openai.api_key = os.environ["OPENAI_API_KEY"]

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "pdf_text_collection"
collection = chroma_client.get_or_create_collection(name=collection_name)

# OpenAI Assistant
from openai import OpenAI
client = OpenAI()

assistant = client.beta.assistants.create(
    name="AI Psychologist",
    instructions="Provide emotional support and stress management advice.",
    tools=[],
    model="gpt-4o-mini",
)
thread = client.beta.threads.create()

# Function to search similar text using ChromaDB
def search_similar_text(query_text, top_k=5):
    """Retrieves similar text from ChromaDB using stored embeddings."""
    print(f"📌 Total stored documents: {collection.count()}")

    results = collection.query(
        query_texts=[query_text],  # Directly use ChromaDB’s stored embeddings
        n_results=top_k,
        include=["metadatas", "documents"]
    )

    print("🔍 RAW RESULTS:", results)  # Debugging

    if "metadatas" in results and results["metadatas"]:
        retrieved_contexts = [meta["text"] for meta in results["metadatas"][0]]
        return "\n".join(retrieved_contexts)
    else:
        return "⚠ No relevant context found!"

# Example Query
query = "Please provide insights on stress management for students."

# Retrieve context from ChromaDB
retrieved_context = search_similar_text(query)

# Print retrieved context
print("🔍 Retrieved Context:")
print(retrieved_context)

# Send context to OpenAI Assistant
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content=f"The following context is retrieved:\n{retrieved_context}\n\n{query}\n Provide only the relevant output."
)

# Stream Assistant Response
class EventHandler(openai.AssistantEventHandler):
    def on_text_created(self, text):
        print("\nAssistant >", end="", flush=True)

    def on_text_delta(self, delta, snapshot):
        print(delta.value, end="", flush=True)

with client.beta.threads.runs.stream(
    thread_id=thread.id,
    assistant_id=assistant.id,
    instructions="Please address the user as Amritha. The user has a premium account.",
    event_handler=EventHandler(),
) as stream:
    stream.until_done()
