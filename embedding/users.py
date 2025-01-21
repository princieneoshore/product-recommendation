from sentence_transformers import SentenceTransformer
import json
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, HasIdCondition
import numpy as np

load_dotenv()
client = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))
USER_COLLECTION_NAME="user_test"
PRODUCT_COLLECTION_NAME="product_test"
file_path = 'users.json'

purchased_weight = 3
added_to_cart_weight = 2
clicked_weight = 1
collection_dimension = 384

def get_product_by_id(id):
    result = client.scroll(
        collection_name=PRODUCT_COLLECTION_NAME,
        scroll_filter=Filter(must=[HasIdCondition(has_id=[id])]),
        with_vectors=True
    )
    return result[0][0]

def get_user_by_id(id):
    result = client.scroll(
        collection_name=USER_COLLECTION_NAME,
        scroll_filter=Filter(must=[HasIdCondition(has_id=[id])]),
        with_vectors=True
    )
    return result[0][0]

def save_users():
    with open(file_path, 'r') as file:
        data = json.load(file)

    for user in data:
        user_vector = np.zeros(collection_dimension)
        for product_id in user['purchased_product_ids']:
            product = get_product_by_id(product_id)
            user_vector = user_vector + purchased_weight * np.array(product.vector)
        for product_id in user['added_to_cart_product_ids']:
            product = get_product_by_id(product_id)
            user_vector = user_vector + added_to_cart_weight * np.array(product.vector)
        for product_id in user['clicked_product_ids']:
            product = get_product_by_id(product_id)
            user_vector = user_vector + clicked_weight * np.array(product.vector)
        magnitude = np.linalg.norm(user_vector)
        user['vector'] = user_vector / magnitude if magnitude != 0 else np.zeros_like(user_vector)

    if not client.collection_exists(USER_COLLECTION_NAME):
        client.create_collection(
            collection_name=USER_COLLECTION_NAME,
            vectors_config=VectorParams(size=collection_dimension, distance=Distance.COSINE),
        )

    client.upsert(
        collection_name=USER_COLLECTION_NAME,
        points=[
            PointStruct(
                    id=user['id'],
                    vector=user['vector'],
                    payload={key: value for key, value in user.items() if key not in {"id", "vector"}}
            )
            for user in data
        ]
    )
    print("users saved in qdrant successfully")


def recommend_products_to_user(user_id):
    user = get_user_by_id(user_id)
    return client.search(
        collection_name=PRODUCT_COLLECTION_NAME,
        query_vector=user.vector,
        limit=5
    )

# save_users()
print(recommend_products_to_user(1))