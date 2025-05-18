from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

model = SentenceTransformer("all-MiniLM-L6-v2")

@app.route("/embed", methods=["POST"])
def embed():
    data = request.get_json()
    vector = model.encode(data["text"], normalize_embeddings=True).tolist()
    return jsonify({"embedding": vector})

if __name__ == "__main__":
    app.run(port=5001)
