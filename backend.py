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

# 🟢 FIX: Increased max_tokens to 1024 to avoid MAX_TOKENS error
def gemini_generate(prompt_text: str, max_tokens: int = 1024) -> str:
    """Call Gemini API to generate text, with robust error and response handling."""
    if not GEMINI_API_KEY:
         return "Error: GEMINI_API_KEY is not set."
         
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "contents": [{
            "parts": [{
                "text": prompt_text
            }]
        }],
        "generationConfig": {
            "maxOutputTokens": max_tokens
        }
    }
    
    try:
        response = requests.post(GEMINI_URL, headers=headers, json=data)
        response.raise_for_status() 
        
        response_json = response.json()
        
        if response_json.get("candidates"):
            candidate = response_json["candidates"][0]
            content = candidate.get("content")
            
            # Safely check for the 'parts' and 'text' keys
            if content and content.get("parts") and content["parts"][0].get("text"):
                return content["parts"][0]["text"]
            else:
                # Handle cases where content is blocked, empty, or incomplete
                finish_reason = candidate.get("finishReason", "UNKNOWN")
                safety_ratings = candidate.get("safetyRatings", "N/A")
                
                # Check for prompt feedback which might explain blocking
                feedback = response_json.get("promptFeedback", "No feedback.")

                return (
                    f"Error: Gemini returned no text content. "
                    f"Finish Reason: {finish_reason}. "
                    f"Safety Ratings: {safety_ratings}. "
                    f"Feedback: {feedback}"
                )
            
        else:
            return f"Error: Gemini returned no candidates. Full response: {response_json}"
            
    except requests.exceptions.HTTPError as http_err:
        print(f"Gemini API HTTP error: {http_err}")
        return f"Error generating response: {http_err}. Full response: {response.text}"
    except Exception as e:
        print(f"Gemini API general error: {e}")
        return f"Error generating response: {str(e)}"

# --- Wrap Gemini API as RunnableLambda ---
gemini_runnable = RunnableLambda(lambda x: gemini_generate(x))

# --- RAG Chain Setup ---
system_prompt = "You are a medical AI assistant. Answer questions based on provided context."
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Context:\n{context}\n\nQuestion: {query}")])

# --- Alternative: Simple approach without RetrievalQA ---
def simple_rag_chain(query):
    """Simplified RAG implementation"""
    if not query:
        return "No query provided"
    
    if retriever is None:
        return "Error: RAG components (Pinecone/Retriever) failed to load."

    try:
        # Get relevant documents
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create the prompt
        messages = prompt_template.format_messages(context=context, query=query)
        
        # Extract the human message content
        prompt_text = messages[-1].content
        
        # Generate response
        return gemini_generate(prompt_text)
    except Exception as e:
        return f"Error processing request: {str(e)}"

# --- API Endpoints ---
@app.route("/get", methods=["POST"])
def chat():
    msg = request.json.get("query")
    if not msg:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Use the simple RAG chain
        response = simple_rag_chain(msg)
        
        if response.startswith("Error"):
            return jsonify({"answer": response}), 500
            
        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return "Medical AI backend is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)