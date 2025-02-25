import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from .db_connector import fetch_interactions

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

# Load models once
user_factors, item_factors, svd_model, product_sim_matrix, df = train_models()

def get_similar_products(product_id, top_n=5):
    product_idx_map = dict(enumerate(df['product_id'].astype('category').cat.categories))
    if product_id not in product_idx_map.values():
        return []
    idx = list(product_idx_map.keys())[list(product_idx_map.values()).index(product_id)]
    scores = product_sim_matrix[idx]
    similar_products = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[1:top_n+1]
    return [product_idx_map[i[0]] for i in similar_products]

def recommend_products(user_id, top_n=5):
    user_idx_map = dict(enumerate(df['user_id'].astype('category').cat.categories))
    product_idx_map = dict(enumerate(df['product_id'].astype('category').cat.categories))

    if user_id not in user_idx_map.values():
        return "User not found"
    
    user_idx = list(user_idx_map.keys())[list(user_idx_map.values()).index(user_id)]
    scores = item_factors @ user_factors[user_idx]
    recommended_idx = np.argsort(scores)[::-1][:top_n]

    return [product_idx_map[i] for i in recommended_idx]

def hybrid_recommendations(user_id, product_id, top_n=5, alpha=0.4, beta=0.6):
    svd_recs = get_similar_products(product_id, top_n)
    als_recs = recommend_products(user_id, top_n)

    combined_scores = {prod: beta for prod in als_recs}
    for prod in svd_recs:
        combined_scores[prod] = combined_scores.get(prod, 0) + alpha

    return sorted(combined_scores, key=combined_scores.get, reverse=True)[:top_n]
