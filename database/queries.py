from .mongo_client import conversation_collection
from bson import ObjectId
from datetime import datetime

def create_conversation(user_id: str):
    convo = {
        "user_id": user_id,
        "created_at": datetime.utcnow(),
        "messages": []
    }
    result = conversation_collection.insert_one(convo)
    convo["_id"] = result.inserted_id
    return convo

def get_all_conversations():
    return list(conversation_collection.find())

def get_conversation_by_id(convo_id: str):
    return conversation_collection.find_one({"_id": ObjectId(convo_id)})

def add_message(convo_id: str, sender: str, content: str):
    msg = {
        "sender": sender,
        "content": content,
        "timestamp": datetime.utcnow()
    }

    if sender == "user":
        bot_reply = {
            "sender": "assistant",
            "content": f'🤖 Bot reply to: "{content}"',
            "timestamp": datetime.utcnow()
        }
        conversation_collection.update_one(
            {"_id": ObjectId(convo_id)},
            {"$push": {"messages": {"$each": [msg, bot_reply]}}}
        )
    else:
        conversation_collection.update_one(
            {"_id": ObjectId(convo_id)},
            {"$push": {"messages": msg}}
        )

    return get_conversation_by_id(convo_id)