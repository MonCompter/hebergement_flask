from flask import Flask, request, jsonify
import clip
import faiss
import torch
import numpy as np
import pickle
from PIL import Image
from io import BytesIO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

index = faiss.read_index("index.faiss")
with open("product_ids.pkl", "rb") as f:
    product_ids = pickle.load(f)
with open("image_paths.pkl", "rb") as f:
    image_paths = pickle.load(f)

@app.route("/search", methods=["POST"])
def search():
    if 'image' not in request.files:
        return jsonify({"error": "Aucune image reçue."}), 400

    file = request.files['image']
    image = Image.open(BytesIO(file.read())).convert("RGB")

    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image_input)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    embedding = embedding.cpu().numpy().astype("float32")

    k = int(request.form.get("k", 5))
    distances, indices = index.search(embedding, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            "product_id": product_ids[idx],
            "distance": float(dist),
            "path": image_paths[idx]
        })

    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
