from sentence_transformers import SentenceTransformer
import json
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
from qdrant_client.models import VectorParams, Distance, PointStruct

load_dotenv()
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and efficient model
client = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))
PRODUCT_COLLECTION_NAME="product_test"
file_path = 'products.json'

def save_products():
    with open(file_path, 'r') as file:
        data = json.load(file)

    for product in data:
        product_text = f"{product['title']} {product['description']} {product['category']}"
        product['vector'] = model.encode(product_text)

    if not client.collection_exists(PRODUCT_COLLECTION_NAME):
        client.create_collection(
            collection_name=PRODUCT_COLLECTION_NAME,
            vectors_config=VectorParams(size=model.get_sentence_embedding_dimension(), distance=Distance.COSINE),
        )

    client.upsert(
        collection_name=PRODUCT_COLLECTION_NAME,
        points=[
            PointStruct(
                    id=product['id'],
                    vector=product['vector'],
                    payload={key: value for key, value in product.items() if key not in {"id", "vector"}} | {"sale_nbr": 0}
            )
            for product in data
        ]
    )
    print("products saved in qdrant successfully")


save_products()