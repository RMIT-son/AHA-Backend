import bcrypt
from bson import ObjectId
from database import Message
from datetime import datetime
from database.schemas import UserCreate, UserLogin
from .mongo_client import conversation_collection, user_collection

# Helper function to convert MongoDB document (_id) into a serializable dictionary
def serialize_mongo_document(doc):
    """Convert MongoDB document to API-friendly format"""
    if not doc:
        return None
    
    doc = doc.copy()
    if "_id" in doc:
        doc["id"] = str(doc["_id"])  # Replace MongoDB's _id with stringified id
        del doc["_id"]
    return doc

# Create a new conversation document in the database
def create_conversation(user_id: str, title: str):
    convo = {
        "title": title,
        "user_id": user_id,
        "created_at": datetime.utcnow(),
        "messages": []
    }
    result = conversation_collection.insert_one(convo)

    # Add the inserted ObjectId as a string id for frontend compatibility
    convo["id"] = str(result.inserted_id)
    
    return convo

# Retrieve all conversation documents and serialize ObjectId to id
def get_all_conversations(user_id: str):
    # Only get conversations belonging to this user
    conversations = list(conversation_collection.find({"user_id": user_id}))
    
    for convo in conversations:
        if "_id" in convo:
            convo["id"] = str(convo["_id"])
            del convo["_id"]
    
    return conversations

# Retrieve a single conversation by its string id
def get_conversation_by_id(convo_id: str):
    try:
        convo = conversation_collection.find_one({"_id": ObjectId(convo_id)})
        if convo:
            return serialize_mongo_document(convo)
        return None
    except Exception as e:
        # If ObjectId is invalid (e.g. wrong format), catch and log
        print(f"Error finding conversation: {e}")
        return None

# Add a user or bot message to an existing conversation
async def add_message(convo_id: str, message: Message, response: str):
    msg = {
        "sender": "user",
        "content": message.content,
        "timestamp": message.timestamp
    }

    bot_reply = {
        "sender": "assistant",
        "content": response,
        "timestamp": datetime.utcnow()
    }

    # Push both user message and bot reply into the conversation
    conversation_collection.update_one(
        {"_id": ObjectId(convo_id)},
        {"$push": {"messages": {"$each": [msg, bot_reply]}}}
    )

def serialize_user(user):
    if not user:
        return None
    return {
        "id": str(user.get("_id", "")),  # ensures string
        "fullName": user.get("fullName", ""),
        "email": user.get("email", ""),
        "phone": user.get("phone", "")
    }


def register_user(user_data: UserCreate):
    print("Registering use function:", user_data)
    existing_user = user_collection.find_one({"email": user_data.email})
    if existing_user:
        raise ValueError("User already exists")

    hashed_pw = bcrypt.hashpw(user_data.password.encode("utf-8"), bcrypt.gensalt())

    new_user = {
        "fullName": user_data.fullName,
        "email": user_data.email,
        "password": hashed_pw.decode("utf-8"),  # Store as string
        "phone": user_data.phone
    }
    print("Create new user", new_user)
    

    result = user_collection.insert_one(new_user)
    print("Inserted user with ID:", result.inserted_id)
    new_user["_id"] = result.inserted_id
    return serialize_user(new_user)


def login_user(credentials: UserLogin):
    user = user_collection.find_one({"email": credentials.email})
    if user and bcrypt.checkpw(credentials.password.encode("utf-8"), user["password"].encode("utf-8")):
        return serialize_user(user)
    return None