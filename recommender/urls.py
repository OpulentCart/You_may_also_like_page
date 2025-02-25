from django.urls import path
from .views import recommend_products_api, similar_products_api, hybrid_recommendations_api

urlpatterns = [
    path("recommend/<int:user_id>/", recommend_products_api, name="recommend-products"),
    path("similar/<int:product_id>/", similar_products_api, name="similar-products"),
    path("hybrid/<int:user_id>/<int:product_id>/", hybrid_recommendations_api, name="hybrid-recommendations"),
]
