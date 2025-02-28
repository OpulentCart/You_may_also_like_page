import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import psycopg2
from psycopg2.extras import RealDictCursor
from django.conf import settings

# Database connection using Django settings
def get_postgres_connection():
    return psycopg2.connect(
        dbname=settings.DATABASES["default"]["NAME"],
        user=settings.DATABASES["default"]["USER"],
        password=settings.DATABASES["default"]["PASSWORD"],
        host=settings.DATABASES["default"]["HOST"],
        port=settings.DATABASES["default"]["PORT"],
    )

def fetch_interactions():
    """Fetch user interactions from the user_interactions table."""
    try:
        with get_postgres_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = """
                    SELECT user_id, product_id, interaction_type, rating, timestamp 
                    FROM user_interactions
                """
                cur.execute(query)
                # Convert to DataFrame for compatibility with existing code
                return pd.DataFrame(cur.fetchall())
    except Exception as e:
        print(f"Error fetching interactions: {e}")
        return pd.DataFrame()

def fetch_product_details(product_ids):
    """Fetch product details from the product table."""
    if not product_ids:
        return []
    try:
        with get_postgres_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = """
                    SELECT product_id AS id, name, brand, price, main_image 
                    FROM product 
                    WHERE product_id IN %s
                """
                cur.execute(query, (tuple(product_ids),))
                return cur.fetchall()  # Returns list of dicts with column names as keys
    except Exception as e:
        print(f"Error fetching product details: {e}")
        return []

def train_als_manual(sparse_matrix, iterations=20, alpha=0.01, factors=50, reg=0.1):
    num_users, num_items = sparse_matrix.shape
    user_factors = np.random.rand(num_users, factors)
    item_factors = np.random.rand(num_items, factors)

    for _ in range(iterations):
        # Update User Factors
        for u in range(num_users):
            relevant_items = sparse_matrix[u].indices
            if len(relevant_items) == 0:
                continue
            item_matrix = item_factors[relevant_items]
            ratings = sparse_matrix[u, relevant_items].toarray().flatten()
            user_factors[u] = np.linalg.solve(item_matrix.T @ item_matrix + reg * np.eye(factors), item_matrix.T @ ratings)

        # Update Item Factors
        for i in range(num_items):
            relevant_users = sparse_matrix[:, i].indices
            if len(relevant_users) == 0:
                continue
            user_matrix = user_factors[relevant_users]
            ratings = sparse_matrix[relevant_users, i].toarray().flatten()
            item_factors[i] = np.linalg.solve(user_matrix.T @ user_matrix + reg * np.eye(factors), user_matrix.T @ ratings)

    return user_factors, item_factors

def train_models():
    df = fetch_interactions()
    if df.empty:
        raise ValueError("No interaction data available to train models.")

    # Define interaction weights
    action_weights = {
        "view": 1,
        "click": 2,
        "add_to_cart": 3,
        "purchase": 5,
        "rating": 4
    }
    df["weight"] = df["interaction_type"].map(action_weights)
    df.loc[df["interaction_type"] == "rating", "weight"] = df["rating"].fillna(0) * 5

    # Time Decay
    lambda_decay = 0.01
    now = pd.Timestamp.now()
    df["days_since"] = (now - pd.to_datetime(df["timestamp"])).dt.days
    df["decay_factor"] = np.exp(-lambda_decay * df["days_since"])
    df["weight"] *= df["decay_factor"]

    # Convert IDs to numerical indices
    df["user_idx"] = df["user_id"].astype("category").cat.codes
    df["product_idx"] = df["product_id"].astype("category").cat.codes

    # Create Sparse Matrix
    rows, cols, values = df["user_idx"], df["product_idx"], df["weight"]
    sparse_matrix = csr_matrix((values, (rows, cols)))

    # Train ALS Model (Manual)
    user_factors, item_factors = train_als_manual(sparse_matrix)

    # Train SVD for item similarity
    n_components = min(50, sparse_matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components)
    svd_matrix = svd.fit_transform(sparse_matrix)

    # Compute Cosine Similarity
    product_sim_matrix = cosine_similarity(svd.components_.T)

    return user_factors, item_factors, svd, product_sim_matrix, df

# Load models once (with error handling)
try:
    user_factors, item_factors, svd_model, product_sim_matrix, df = train_models()
except ValueError as e:
    print(f"Training failed: {e}")
    user_factors, item_factors, svd_model, product_sim_matrix, df = None, None, None, None, pd.DataFrame()

def get_similar_products(product_id, top_n=5):
    if product_sim_matrix is None or df.empty:
        return []
    
    product_idx_map = dict(enumerate(df['product_id'].astype('category').cat.categories))
    if product_id not in product_idx_map.values():
        return []
    
    idx = list(product_idx_map.keys())[list(product_idx_map.values()).index(product_id)]
    scores = product_sim_matrix[idx]
    similar_products = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[1:top_n+1]
    product_ids = [product_idx_map[i[0]] for i in similar_products]
    
    # Fetch product details
    product_details = fetch_product_details(product_ids)
    # Add similarity scores
    for i, detail in enumerate(product_details):
        detail["similarity_score"] = float(scores[similar_products[i][0]])
    
    return product_details

def recommend_products(user_id, top_n=5):
    if user_factors is None or item_factors is None or df.empty:
        return "Model not trained"
    
    user_idx_map = dict(enumerate(df['user_id'].astype('category').cat.categories))
    product_idx_map = dict(enumerate(df['product_id'].astype('category').cat.categories))

    if user_id not in user_idx_map.values():
        return "User not found"
    
    user_idx = list(user_idx_map.keys())[list(user_idx_map.values()).index(user_id)]
    scores = item_factors @ user_factors[user_idx]
    recommended_idx = np.argsort(scores)[::-1][:top_n]
    product_ids = [product_idx_map[i] for i in recommended_idx]
    
    # Fetch product details
    product_details = fetch_product_details(product_ids)
    # Add recommendation scores (normalized for display)
    max_score = scores.max() if scores.max() > 0 else 1
    for i, detail in enumerate(product_details):
        detail["recommendation_score"] = float(scores[recommended_idx[i]] / max_score)
    
    return product_details

def hybrid_recommendations(user_id, product_id, top_n=5, alpha=0.4, beta=0.6):
    if user_factors is None or product_sim_matrix is None or df.empty:
        return []
    
    # Get recommendations from both models
    svd_recs = get_similar_products(product_id, top_n * 2)  # Get extra to allow blending
    als_recs = recommend_products(user_id, top_n * 2) if isinstance(recommend_products(user_id), list) else []

    # Combine scores using product_id as the key
    combined_scores = {}
    for rec in als_recs:
        combined_scores[rec["id"]] = {"score": beta * rec["recommendation_score"], "details": rec}
    for rec in svd_recs:
        pid = rec["id"]
        if pid in combined_scores:
            combined_scores[pid]["score"] += alpha * rec["similarity_score"]
        else:
            combined_scores[pid] = {"score": alpha * rec["similarity_score"], "details": rec}

    # Sort and select top_n
    sorted_products = sorted(combined_scores.items(), key=lambda x: x[1]["score"], reverse=True)[:top_n]
    result = [item[1]["details"] for item in sorted_products]
    for i, res in enumerate(result):
        res["hybrid_score"] = float(sorted_products[i][1]["score"])
        # Remove intermediate scores to match desired output
        res.pop("recommendation_score", None)
        res.pop("similarity_score", None)
    
    return result

# Django views
from django.http import JsonResponse

def similar_products_view(request, product_id):
    result = get_similar_products(product_id)
    return JsonResponse({"related_products": result})

def recommendations_view(request, user_id):
    result = recommend_products(user_id)
    return JsonResponse({"recommended_products": result})

def hybrid_recommendations_view(request, user_id, product_id):
    result = hybrid_recommendations(user_id, product_id)
    return JsonResponse({"hybrid_recommendations": result})