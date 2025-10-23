import os
import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify

# --- LangChain imports ---
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda

# --- Load environment variables ---
load_dotenv()

app = Flask(__name__)

# =====================
# --- Pinecone Setup ---
# =====================
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "medicalbot"
PINECONE_ENVIRONMENT = "us-east1-gcp"  # Replace with your Pinecone environment

# Set API key for Pinecone
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

try:
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Initialize PineconeVectorStore directly
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )

    # Create retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    print("✅ Pinecone index connected successfully.")
except Exception as e:
    print(f"❌ Error loading Pinecone index: {e}")
    retriever = None

# =====================
# --- Gemini API Setup ---
# =====================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"

def gemini_generate(prompt_text: str, max_tokens: int = 1024) -> str:
    """Call Gemini API to generate text."""
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY is not set."

    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {"maxOutputTokens": max_tokens}
    }

    try:
        response = requests.post(GEMINI_URL, headers=headers, json=data)
        response.raise_for_status()
        response_json = response.json()

        if response_json.get("candidates"):
            candidate = response_json["candidates"][0]
            content = candidate.get("content")
            if content and content.get("parts") and content["parts"][0].get("text"):
                return content["parts"][0]["text"]
            else:
                return f"Error: Gemini returned no text. Finish reason: {candidate.get('finishReason', 'UNKNOWN')}"
        else:
            return f"Error: Gemini returned no candidates. Full response: {response_json}"
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Wrap Gemini in RunnableLambda for LangChain usage
gemini_runnable = RunnableLambda(lambda x: gemini_generate(x))

# =====================
# --- RAG Prompt Setup ---
# =====================
system_prompt = "You are a medical AI assistant. Answer questions based on provided context."
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Context:\n{context}\n\nQuestion: {query}")
])

def simple_rag_chain(query):
    """Retrieve documents from Pinecone and generate answer via Gemini."""
    if not query:
        return "No query provided."

    if retriever is None:
        return "Error: Pinecone retriever failed to load."

    try:
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        messages = prompt_template.format_messages(context=context, query=query)
        prompt_text = messages[-1].content
        return gemini_generate(prompt_text)
    except Exception as e:
        return f"Error processing request: {str(e)}"

# =====================
# --- API Endpoints ---
# =====================
@app.route("/get", methods=["POST"])
def chat():
    msg = request.json.get("query")
    if not msg:
        return jsonify({"error": "No message provided"}), 400

    response = simple_rag_chain(msg)
    if response.startswith("Error"):
        return jsonify({"answer": response}), 500

    return jsonify({"answer": response})

@app.route("/")
def index():
    return "✅ Medical AI backend is running!"

# =====================
# --- Run Flask ---
# =====================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Render dynamic port
    app.run(host="0.0.0.0", port=port)
