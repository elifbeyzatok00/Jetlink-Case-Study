from flask import Flask, request, jsonify, session
from flask_pymongo import PyMongo
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")

# Configure MongoDB
app.config["MONGO_URI"] = os.getenv("MONGO_URI")
mongo = PyMongo(app)

# Start Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Index control and creation
index_name = "jetlink-db"
if index_name in pc.list_indexes().names():
    print(f"Index {index_name} already exists.")
elif index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384, # Embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"Index {index_name} created.")

# Connect to vector store
index = pc.Index(index_name)

# Log in using your Hugging Face token
login(token=os.getenv("HUGGING_FACE_TOKEN"))

# Available LLM models
LLM_MODELS = {
    "LLAMA_3_2_1B_INSTURCT": "meta-llama/Llama-3.2-1B-Instruct",
    "LLAMA_3_2_3B_INSTURCT": "meta-llama/Llama-3.2-3B-Instruct"
}

def load_llm(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# Default LLM
current_model = os.getenv("DEFAULT_LLM", "LLAMA_3_2_1B_INSTURCT")
tokenizer, model = load_llm(LLM_MODELS[current_model])

# Load Embedding Model
embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
embedding_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")

def get_embedding(text):
    inputs = embedding_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.numpy().tolist()[0]

# Helper function to store long-term memory
def store_long_term_memory(user_id, message, response):
        # Metni embedding vektörüne çevir
    vector = get_embedding(message)

    # Pinecone'a ekleme yap
    index.upsert(vectors=[
        {"id": str(user_id) + "_" + str(hash(message)), "values": vector, "metadata": {"message": message, "response": response}}
    ])
    
    # MongoDB'ye ekleme yap
    mongo.db.memory.insert_one({"user_id": user_id, "message": message, "response": response})

# Helper function for vector search
def search_memory(query):
    vector = get_embedding(query)
    results = index.query(vector=vector, top_k=3, include_metadata=True)
    return results

@app.route("/")
def home():
    return "Flask Chatbot API is running!", 200


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_id = data.get("user_id", "anonymous")
    message = data.get("message", "")

    # Retrieve session-based short-term memory
    session_memory = session.get("history", [])
    
    # Retrieve relevant long-term memory
    memory_results = search_memory(message)

    # Construct context
    context = "\n".join([item["metadata"]["response"] for item in memory_results["matches"]])
    context += "\n".join(session_memory)

    # Generate response
    inputs = tokenizer(context + "\nUser: " + message + "\nBot:", return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_length=3000)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    
    # Update memory
    session_memory.append(message)
    session_memory.append(response)
    session["history"] = session_memory[-10:]  # Keep last 10 messages
    store_long_term_memory(user_id, message, response)
    
    return jsonify({"response": response})

@app.route("/clear_memory", methods=["POST"])
def clear_memory():
    user_id = request.json.get("user_id", "anonymous")
    mongo.db.memory.delete_many({"user_id": user_id})
    session.pop("history", None)
    return jsonify({"message": "Memory cleared successfully"})

@app.route("/get_llms", methods=["GET"])
def get_llms():
    """Mevcut LLM modellerini döndürür."""
    return jsonify({"models": list(LLM_MODELS.keys())})

@app.route("/set_llm", methods=["POST"])
def set_llm():
    global tokenizer, model
    data = request.json
    selected_model = data.get("model")
    if selected_model in LLM_MODELS:
        tokenizer, model = load_llm(LLM_MODELS[selected_model])
        return jsonify({"message": f"LLM changed to {selected_model}"})
    return jsonify({"error": "Invalid model selection"}), 400

if __name__ == "__main__":
    app.run(debug=True)
