from flask import Flask, request, jsonify
import os
import requests
from dotenv import load_dotenv

# LangChain 1.0+ imports
from langchain.chains import RetrievalQA
from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.vectorstores import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnableLambda

load_dotenv()
app = Flask(__name__)

# --- Pinecone Setup ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
index_name = "medicalbot"

# --- Load embeddings ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Load existing Pinecone index ---
try:
    docsearch = Pinecone.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    print("Pinecone index loaded successfully.")
except Exception as e:
    print(f"Error loading Pinecone index: {e}")
    retriever = None

# --- Gemini API Setup ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash" 
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"

def gemini_generate(prompt_text: str, max_tokens: int = 1024) -> str:
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
        resp_json = response.json()
        if resp_json.get("candidates"):
            candidate = resp_json["candidates"][0]
            content = candidate.get("content")
            if content and content.get("parts") and content["parts"][0].get("text"):
                return content["parts"][0]["text"]
            else:
                finish_reason = candidate.get("finishReason", "UNKNOWN")
                safety_ratings = candidate.get("safetyRatings", "N/A")
                feedback = resp_json.get("promptFeedback", "No feedback.")
                return f"Error: Gemini returned no text content. Finish Reason: {finish_reason}. Safety Ratings: {safety_ratings}. Feedback: {feedback}"
        else:
            return f"Error: Gemini returned no candidates. Full response: {resp_json}"
    except requests.exceptions.HTTPError as http_err:
        return f"Error generating response: {http_err}. Full response: {response.text}"
    except Exception as e:
        return f"Error generating response: {str(e)}"

gemini_runnable = RunnableLambda(lambda x: gemini_generate(x))

system_prompt = "You are a medical AI assistant. Answer questions based on provided context."
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Context:\n{context}\n\nQuestion: {query}")
])

def simple_rag_chain(query):
    if not query:
        return "No query provided"
    if retriever is None:
        return "Error: RAG components (Pinecone/Retriever) failed to load."
    try:
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        messages = prompt_template.format_messages(context=context, query=query)
        prompt_text = messages[-1].content
        return gemini_generate(prompt_text)
    except Exception as e:
        return f"Error processing request: {str(e)}"

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
    return "Medical AI backend is running!"

if __name__ == "__main__":
    # 🔹 IMPORTANT: Use Render's PORT environment variable
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
