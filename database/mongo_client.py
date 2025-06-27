from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

client = MongoClient(os.getenv("MONGO_DB_URL"))
db = client["AHA"]
conversation_collection = db["conversations"]
user_collection = db["users"]

# Test connection
try:
    # Ping the database
    client.admin.command('ping')
    print("Successfully connected to MongoDB Atlas!")
except Exception as e:
    print(f"Connection failed: {e}")