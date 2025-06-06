from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

# Debug: Print the connection string (remove this after testing)
print("MONGO_DB_URL:", os.getenv("MONGO_DB_URL"))

client = MongoClient(os.getenv("MONGO_DB_URL"))
db = client["AHA-Capstone"]
conversation_collection = db["conversations"]

# Test connection
try:
    # Ping the database
    client.admin.command('ping')
    print("Successfully connected to MongoDB Atlas!")
except Exception as e:
    print(f"Connection failed: {e}")