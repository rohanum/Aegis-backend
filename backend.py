from flask import Flask, request, jsonify
import os
import requests
from dotenv import load_dotenv

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnableLambda
from langchain.prompts.chat import ChatPromptTemplate

# Pinecone v2 client
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
app = Flask(__name__)

# --- Pinecone Setup ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalbot"

# ✅ Ensure the index exists
if index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,  # match your embeddings dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2")
    )

# Now get the index object
index = pc.Index(index_name)

# --- Embeddings ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Retriever ---
from langchain.vectorstores import Pinecone as LC_Pinecone
docsearch = LC_Pinecone(index=index, embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# --- Gemini API Setup ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"

def gemini_generate(prompt_text: str, max_tokens: int = 1024) -> str:
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY is not set."
    headers = {"Content-Type": "application/json"}
    data = {"contents":[{"parts":[{"text":prompt_text}]}],"generationConfig":{"maxOutputTokens":max_tokens}}
    try:
        resp = requests.post(GEMINI_URL, headers=headers, json=data)
        resp.raise_for_status()
        result = resp.json()
        candidates = result.get("candidates")
        if candidates:
            content = candidates[0].get("content", {})
            return content.get("parts", [{}])[0].get("text", "No text returned.")
        return f"Error: No candidates returned. {result}"
    except Exception as e:
        return f"Gemini API error: {str(e)}"

gemini_runnable = RunnableLambda(lambda x: gemini_generate(x))

system_prompt = "You are a medical AI assistant. Answer questions based on provided context."
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Context:\n{context}\n\nQuestion: {query}")
])

def simple_rag_chain(query):
    if not query:
        return "No query provided"
    try:
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([d.page_content for d in docs])
        messages = prompt_template.format_messages(context=context, query=query)
        return gemini_generate(messages[-1].content)
    except Exception as e:
        return f"Error processing request: {str(e)}"

@app.route("/get", methods=["POST"])
def chat():
    msg = request.json.get("query")
    if not msg:
        return jsonify({"error": "No message provided"}), 400
    return jsonify({"answer": simple_rag_chain(msg)})

@app.route("/")
def index():
    return "Medical AI backend is running!"

if __name__ == "__main__":
    # ✅ Use Render PORT
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
