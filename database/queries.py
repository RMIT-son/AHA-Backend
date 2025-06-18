from .mongo_client import conversation_collection
from bson import ObjectId
from datetime import datetime
from database import Message

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
def create_conversation(user_id: str):
    convo = {
        "user_id": user_id,
        "created_at": datetime.utcnow(),
        "messages": []
    }
    result = conversation_collection.insert_one(convo)

    # Add the inserted ObjectId as a string id for frontend compatibility
    convo["id"] = str(result.inserted_id)
    
    return convo

# Retrieve all conversation documents and serialize ObjectId to id
def get_all_conversations():
    conversations = list(conversation_collection.find())
    
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
# If the sender is "user", also generate and store the bot response
async def add_message(convo_id: str, message: Message, response: str):
    msg = {
        "sender": message.sender,
        "content": message.content,
        "timestamp": datetime.utcnow()
    }

    if message.sender == "user":

        # Format bot reply
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
    else:
        # Just store the message (likely system or assistant message)
        conversation_collection.update_one(
            {"_id": ObjectId(convo_id)},
            {"$push": {"messages": msg}}
        )
