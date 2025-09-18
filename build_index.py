import os
import faiss
import clip
import torch
import numpy as np
from PIL import Image
import pickle
import pymysql
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Connexion DB
conn = pymysql.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USERNAME"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_DATABASE")
)
cursor = conn.cursor()

# CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

embeddings = []
product_ids = []
image_paths = []

base_laravel_storage = r"C:\xampp\htdocs\bagisto-2.2\bagisto\storage\app\public"
print(f"Chemin base Laravel storage utilisé : {base_laravel_storage}")

cursor.execute("SELECT path, product_id FROM bproduct_images")
rows = cursor.fetchall()

for path, product_id in rows:
    image_path = os.path.join(base_laravel_storage, path)
    if not os.path.isfile(image_path):
        print(f"[ERREUR] Fichier introuvable : {image_path}")
        continue
    try:
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image_input)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        embeddings.append(embedding.cpu().numpy())
        product_ids.append(product_id)
        image_paths.append(path)  # On ajoute le chemin relatif ici
        print(f"[OK] {path} → produit {product_id}")
    except Exception as e:
        print(f"[ERREUR] {path} : {e}")

if not embeddings:
    print("❌ Aucun embedding généré.")
    exit()

array = np.vstack(embeddings).astype("float32")
index = faiss.IndexFlatL2(array.shape[1])
index.add(array)

faiss.write_index(index, "index.faiss")
with open("product_ids.pkl", "wb") as f:
    pickle.dump(product_ids, f)
with open("image_paths.pkl", "wb") as f:
    pickle.dump(image_paths, f)

print("✅ Index FAISS, product_ids.pkl et image_paths.pkl générés.")
