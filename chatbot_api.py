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

# Load LLM model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

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
        output = model.generate(**inputs, max_length=2000)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # def edit_response(response):
    #     """_summary_
    #     Remove the "User:" and "Bot:" prefixes and unwanted text sections from the response. 
    #     """
    #     # Find the first "Bot:" index and the second "User:" index
    #     bot_start_index = response.find("Bot:") + len("Bot:")  # start index for the Bot response
    #     user_second_start_index = response.find("User:", bot_start_index)  # find the second "User:"

    #     # If no second "User:" is found, take until the end of the response
    #     response_end_index = user_second_start_index if user_second_start_index != -1 else len(response)

    #     # Extract the section of the response after the first "Bot:" and before the second "User:"
    #     final_response = response[bot_start_index:response_end_index].strip()
    #     return final_response
    
    # response = edit_response(response)
    
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



if __name__ == "__main__":
    app.run(debug=True)
