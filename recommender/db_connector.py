import psycopg2
import pandas as pd
from django.conf import settings

# PostgreSQL Database Connection
def get_db_connection():
    return psycopg2.connect(
        dbname=settings.DATABASES["default"]["NAME"],
        user=settings.DATABASES["default"]["USER"],
        password=settings.DATABASES["default"]["PASSWORD"],
        host=settings.DATABASES["default"]["HOST"],
        port=settings.DATABASES["default"]["PORT"]
    )

# Fetch data from PostgreSQL
def fetch_interactions():
    query = "SELECT user_id, product_id, interaction_type, rating, timestamp FROM user_interactions"
    
    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn)
    
    return df

if __name__ == "__main__":
    interactions_df = fetch_interactions()
    print(interactions_df.head())
