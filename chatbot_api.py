from flask import Flask, request, jsonify, session, Response
from flask_pymongo import PyMongo
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, BitsAndBytesConfig
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
from huggingface_hub import login
import torch

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

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 4-bit quantization
    bnb_4bit_compute_dtype=torch.float16,  # Düşük hassasiyetli hesaplama
    bnb_4bit_use_double_quant=True  # Çift quantization ile hız artışı
)

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_llm(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config).to(device)
    return tokenizer, model

# Default LLM
current_model = os.getenv("DEFAULT_LLM", "LLAMA_3_2_1B_INSTURCT")
tokenizer, model = load_llm(LLM_MODELS[current_model])

if torch.cuda.is_available():
    print("CUDA is available")
    print(torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")

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
        {"id": str(hash(message)), "values": vector, "metadata": {"message": message, "response": response, "user_id": user_id}}
    ])
    
    # MongoDB'ye ekleme yap
    mongo.db.memory.insert_one({"user_id": user_id, "message": message, "response": response})

# Search vector memory
def search_memory(query):
    vector = get_embedding(query)
    results = index.query(vector=vector, top_k=3, include_metadata=True)
    return results

# Edit response function to clean up the response
def edit_response(response):
    """Remove the 'User:' and 'Bot:' prefixes and unwanted text sections."""
    bot_start_index = response.find("Bot:") + len("Bot:")  # start index for the Bot response
    user_second_start_index = response.find("User:", bot_start_index)  # find the second "User:"
    response_end_index = user_second_start_index if user_second_start_index != -1 else len(response)
    final_response = response[bot_start_index:response_end_index].strip()
    return final_response

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
    inputs = tokenizer(context + "\nUser: " + message + "\nBot:", return_tensors="pt").to(device)
    
    
    # Streaming generator function
    def generate_response_stream():
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=1024, temperature=0.7, top_p=0.9, do_sample=True)
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            cleaned_response = edit_response(response)  # Clean the response using the edit_response function
            yield cleaned_response
    
    # Update memory
    response = next(generate_response_stream())
    session_memory.append(message)
    session_memory.append(response)
    session["history"] = session_memory[-10:]  # Keep last 10 messages
    store_long_term_memory(user_id, message, response)
    
    return jsonify({"response": response})

@app.route("/clear_memory", methods=["POST"])
def clear_memory():
    user_id = request.json.get("user_id", "anonymous")
    # MongoDB temizle
    mongo.db.memory.delete_many({"user_id": user_id})
    
    # Session temizle 
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
        return jsonify({"message": f"LLM changed to {selected_model}"}), 200
    return jsonify({"error": "Invalid model selection"}), 400

if __name__ == "__main__":
    app.run(debug=True)
