import os
import requests
import chromadb
from chromadb.utils import embedding_functions
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from openai import OpenAI
from collections import Counter
import database  # Your existing database module

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# YouTube API Key (Replace with your actual key)
YOUTUBE_API_KEY = "AIzaSyAB1N-DNW-aNldjDrKqmoQglqxM_i594wg"

# Define embedding function using SentenceTransformers
EMBED_MODEL = "all-MiniLM-L6-v2"
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_recommendation_db")

# Define ChromaDB collection with embedding function
collection_name = "pdf_keywords_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=embedding_func
)

# Initialize OpenAI client for additional analysis
try:
    openai_api_key = os.environ.get("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")
    if openai_api_key:
        openai_client = OpenAI(api_key=openai_api_key)
    else:
        openai_client = None
        print("⚠ OpenAI API key not found. Some advanced features will be disabled.")
except:
    openai_client = None
    print("⚠ OpenAI integration disabled.")

# Predefined PDF path
PDF_PATH = "The_Stress_Management.pdf"  # <-- Hardcoded PDF path

def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    try:
        with open(pdf_file, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text.strip()
    except Exception as e:
        print(f"❌ Error extracting text from PDF: {e}")
        return ""

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

def extract_keywords(text, top_n=20):
    """Extract most important keywords from text using TF-IDF approach."""
    words = nltk.word_tokenize(text.lower())
    # Remove stopwords and non-alphabetic words, then lemmatize
    filtered_words = [lemmatizer.lemmatize(word) for word in words
                      if word.isalpha() and word not in stop_words and len(word) > 2]

    # Count word frequencies
    word_freq = Counter(filtered_words)

    # Get top N keywords
    keywords = [word for word, _ in word_freq.most_common(top_n)]

    return keywords

def analyze_with_openai(text, user_interests):
    """Use OpenAI to extract keywords and themes relevant to user interests."""
    if not openai_client:
        return []

    try:
        prompt = f"""
        Analyze the following text and extract 5-10 key concepts related to stress management 
        that would be most relevant to someone interested in: {user_interests}.
        Focus on actionable topics that could be searched on YouTube.
        Text: {text[:4000]}...

        Provide the output as a simple comma-separated list of keywords/phrases.
        """

        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )

        # Parse the comma-separated response
        keywords_text = response.choices[0].message.content.strip()
        ai_keywords = [k.strip() for k in keywords_text.split(',')]

        return ai_keywords
    except Exception as e:
        print(f"⚠ OpenAI analysis failed: {e}")
        return []

def index_pdf(pdf_file, pdf_id):
    """Index a PDF file into ChromaDB with extracted keywords."""
    pdf_text = extract_text_from_pdf(pdf_file)
    if not pdf_text:
        return False

    # Split text into chunks
    text_chunks = split_text(pdf_text)

    # Process and store each chunk
    for i, chunk in enumerate(text_chunks):
        # Extract keywords from this chunk
        keywords = extract_keywords(chunk)

        # Create metadata with text and keywords
        metadata = {
            "pdf_id": pdf_id,
            "chunk_id": i,
            "text": chunk[:1000],  # Store partial text for context
            "keywords": ", ".join(keywords)
        }

        # Add to ChromaDB
        collection.add(
            ids=[f"{pdf_id}-chunk-{i}"],
            documents=[chunk],
            metadatas=[metadata]
        )

    print(f"✅ Successfully indexed PDF with ID {pdf_id}")
    return True

def retrieve_relevant_content(query, user_interests, top_k=5):
    """Retrieve relevant content from ChromaDB based on query and user interests."""
    # Combine query with user interests for better context
    enhanced_query = f"{query} {user_interests}"

    # Query the collection
    results = collection.query(
        query_texts=[enhanced_query],
        n_results=top_k,
        include=["metadatas", "documents"]
    )

    if not results["metadatas"][0]:
        return [], []

    # Extract documents and keywords
    relevant_texts = results["documents"][0]

    # Collect all keywords from metadata
    all_keywords = []
    for metadata in results["metadatas"][0]:
        if "keywords" in metadata:
            all_keywords.extend(metadata["keywords"].split(", "))

    # Count keyword frequencies across all retrieved documents
    keyword_counts = Counter(all_keywords)

    # Get top keywords
    top_keywords = [keyword for keyword, _ in keyword_counts.most_common(10)]

    return relevant_texts, top_keywords

def fetch_youtube_videos(search_query, max_results=3):
    """Fetch YouTube videos based on a search query."""
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={search_query}&key={YOUTUBE_API_KEY}&maxResults={max_results}&type=video"

    try:
        response = requests.get(url).json()

        video_links = []
        for item in response.get("items", []):
            video_id = item["id"]["videoId"]
            title = item["snippet"]["title"]
            video_links.append({
                "title": title,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"]
            })

        return video_links
    except Exception as e:
        print(f"❌ Error fetching YouTube videos: {e}")
        return []

def generate_recommendations(user_interests, stress_level):
    """Generate video recommendations based on user interests and stress level."""
    # Retrieve relevant content from indexed PDFs
    relevant_texts, keywords = retrieve_relevant_content(
        f"stress management {stress_level}",
        user_interests
    )

    # If we have OpenAI available, enhance our keywords with AI analysis
    if openai_client and relevant_texts:
        combined_text = " ".join(relevant_texts[:3])  # Limit to first 3 texts
        ai_keywords = analyze_with_openai(combined_text, user_interests)
        # Combine AI keywords with our extracted keywords
        keywords = list(set(keywords + ai_keywords))

    # Prepare search queries
    search_queries = []
    interests_list = [interest.strip() for interest in user_interests.split(",")]

    # Create targeted search queries combining stress level, keywords, and interests
    for interest in interests_list:
        # Create a specific query for this interest
        query = f"stress relief {stress_level} {interest}"
        if keywords:
            # Add some of the top keywords to make the query more specific
            query_keywords = " ".join(keywords[:3])  # Use top 3 keywords
            query = f"{query} {query_keywords}"

        search_queries.append(query)

    # Fetch videos for each query
    all_videos = []
    for query in search_queries:
        videos = fetch_youtube_videos(query, max_results=2)  # Limit to 2 per interest for diversity
        all_videos.extend(videos)

    # Remove duplicates (if any videos appear in multiple queries)
    unique_videos = []
    seen_urls = set()
    for video in all_videos:
        if video["url"] not in seen_urls:
            unique_videos.append(video)
            seen_urls.add(video["url"])

    return unique_videos[:5]  # Return top 5 unique videos

def conduct_pss_survey():
    """Conducts the PSS (Perceived Stress Scale) survey."""
    questions = [
        "⿡ In the last month, how often have you felt upset because of something unexpected?",
        "⿢ In the last month, how often have you felt unable to control important things in your life?",
        "⿣ In the last month, how often have you felt nervous or stressed?",
        "⿤ In the last month, how often have you felt confident in handling personal problems?",
        "⿥ In the last month, how often have you felt things were going your way?",
        "⿦ In the last month, how often have you found it difficult to cope?",
        "⿧ In the last month, how often have you been able to control irritations?",
        "⿨ In the last month, how often have you felt on top of things?",
        "⿩ In the last month, how often have you been angered by things outside your control?",
        "🔟 In the last month, how often have you felt difficulties piling up?"
    ]

    print("\n🧠 PSS Stress Survey")
    print("Rate the following from 0 (Never) to 4 (Very Often):\n")

    score = 0
    for question in questions:
        while True:
            try:
                response = int(input(f"{question} (0-4): "))
                if 0 <= response <= 4:
                    score += response
                    break
                else:
                    print("❌ Please enter a number between 0 and 4.")
            except ValueError:
                print("❌ Invalid input. Enter a number between 0 and 4.")

    # Categorizing stress levels
    if score <= 13:
        stress_level = "Low"
    elif 14 <= score <= 26:
        stress_level = "Moderate"
    else:
        stress_level = "High"

    return stress_level

def main():
    # Initialize database
    database.init_db()

    print("\n👤 Welcome to the Stress Management System")
    print("\n⿡ Register\n⿢ Login\n⿣ Index PDF")

    option = input("Select an option (1, 2, or 3): ")

    if option == "1":
        # Registration logic
        username = input("Enter a new username: ")
        password = input("Enter a password: ")

        print("\n🎯 Select your area of interest:")
        interests_list = ["Music", "Exercise", "Reading", "Meditation", "Gaming", "Art", "Cooking"]

        for idx, interest in enumerate(interests_list, 1):
            print(f"{idx}. {interest}")

        selected_indices = input("Enter the numbers corresponding to your interests (comma-separated): ")

        try:
            selected_interests = [interests_list[int(i.strip()) - 1] for i in selected_indices.split(",") if
                                  i.strip().isdigit()]
        except (IndexError, ValueError):
            print("❌ Invalid selection. Please enter valid numbers.")
            return

        interests_str = ", ".join(selected_interests)

        if database.register_user(username, password, interests_str):
            print(f"✅ Registration successful! You can now log in as {username}.")
        else:
            print("❌ Username already exists. Try a different one.")

    elif option == "2":
        # Login logic
        username = input("Enter your username: ")
        password = input("Enter your password: ")

        user = database.authenticate_user(username, password)

        if user:
            print(f"\n✅ Welcome back, {username}!")
            print(f"🎯 Your interests: {user[3]}")

            # Conduct stress survey
            stress_level = conduct_pss_survey()
            print(f"\n🧘 Your stress level: {stress_level}")

            # Generate recommendations using RAG approach
            video_results = generate_recommendations(user[3], stress_level)

            print("\n🎥 Top YouTube Videos for Stress Relief Based on Your Profile & RAG Analysis:")
            for video in video_results:
                print(f"- {video['title']}: {video['url']}")
        else:
            print("❌ Invalid credentials. Please try again.")

    elif option == "3":
        # PDF indexing logic
        pdf_id = input("Enter a unique ID for this PDF: ")  # <-- Only ask for the PDF ID

        if os.path.exists(PDF_PATH):  # <-- Use the predefined PDF path
            success = index_pdf(PDF_PATH, pdf_id)  # <-- Pass the predefined PDF path
            if success:
                print(f"✅ Successfully indexed PDF '{PDF_PATH}' with ID '{pdf_id}'")
            else:
                print(f"❌ Failed to index PDF '{PDF_PATH}'")
        else:
            print(f"❌ PDF file not found at path: {PDF_PATH}")


if __name__ == "__main__":
    main()
