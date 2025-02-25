from django.http import JsonResponse
from .recommender import recommend_products, get_similar_products, hybrid_recommendations
from django.http import JsonResponse
from django.db import connection
from django.views.decorators.csrf import csrf_exempt 

import json

def recommend_products_api(request, user_id):
    recommendations = recommend_products(user_id)
    return JsonResponse({"user_id": user_id, "recommended_products": recommendations})

def similar_products_api(request, product_id):
    recommendations = get_similar_products(product_id)
    return JsonResponse({"product_id": product_id, "similar_products": recommendations})

def hybrid_recommendations_api(request, user_id, product_id):
    recommendations = hybrid_recommendations(user_id, product_id)
    return JsonResponse({"user_id": user_id, "product_id": product_id, "recommended_products": recommendations})


@csrf_exempt
def add_user_interaction(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            user_id = data.get("user_id")
            product_id = data.get("product_id")
            interaction_type = data.get("interaction_type")
            rating = data.get("rating")  # Can be None
           
            # Validate input
            if not all([user_id, product_id, interaction_type]):
                return JsonResponse({"error": "Missing required fields"}, status=400)
            
            if interaction_type not in ["view", "click", "add_to_cart", "purchase", "rating"]:
                return JsonResponse({"error": "Invalid interaction type"}, status=400)

            if rating is not None and (rating < 0 or rating > 5):
                return JsonResponse({"error": "Rating must be between 0 and 5"}, status=400)
            
            # Insert data into the database
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO user_interactions (user_id, product_id, interaction_type, rating) 
                    VALUES (%s, %s, %s, %s)
                    """,
                    [user_id, product_id, interaction_type, rating]
                )

            return JsonResponse({"message": "Interaction added successfully"}, status=201)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Only POST method is allowed"}, status=405)
