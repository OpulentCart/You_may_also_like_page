from django.http import JsonResponse
from .recommender import recommend_products, get_similar_products, hybrid_recommendations

def recommend_products_api(request, user_id):
    recommendations = recommend_products(user_id)
    return JsonResponse({"user_id": user_id, "recommended_products": recommendations})

def similar_products_api(request, product_id):
    recommendations = get_similar_products(product_id)
    return JsonResponse({"product_id": product_id, "similar_products": recommendations})

def hybrid_recommendations_api(request, user_id, product_id):
    recommendations = hybrid_recommendations(user_id, product_id)
    return JsonResponse({"user_id": user_id, "product_id": product_id, "recommended_products": recommendations})
