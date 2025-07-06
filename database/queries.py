import bcrypt
from typing import Dict
from bson import ObjectId
from datetime import datetime
from fastapi import HTTPException
from app.utils import build_error_response, serialize_mongo_document
from database.schemas import UserCreate, UserLogin, Message
from .mongo_client import conversation_collection, user_collection
from database.qdrant_client import add_message_vector, delete_conversation_vectors

# Create a new conversation document in the database
def create_conversation(user_id: str, title: str):
    """
    Create a new conversation document for a given user.

    Args:
        user_id (str): The ID of the user who owns the conversation.
        title (str): The title of the conversation.

    Returns:
        dict: The newly created conversation document with an `id` field.
    
    Raises:
        JSONResponse: Error response if creation fails.
    """
    try:
        if not user_id or not title:
            return build_error_response(
                "INVALID_INPUT",
                "User ID and title are required",
                400
            )
        
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
    except Exception as e:
        return build_error_response(
            "CONVERSATION_CREATION_FAILED",
            f"Failed to create conversation: {str(e)}",
            500
        )

# Retrieve all conversation documents and serialize ObjectId to id
def get_all_conversations(user_id: str):
    """
    Retrieve all conversations belonging to a specific user.

    Args:
        user_id (str): User ID to filter conversations.

    Returns:
        list: A list of serialized conversation documents.
    
    Raises:
        JSONResponse: Error response if retrieval fails.
    """
    try:
        if not user_id:
            return build_error_response(
                "INVALID_INPUT",
                "User ID is required",
                400
            )
        
        # Only get conversations belonging to this user
        conversations = list(conversation_collection.find({"user_id": user_id}))
        
        for convo in conversations:
            if "_id" in convo:
                convo["id"] = str(convo["_id"])
                del convo["_id"]
        
        return conversations
    except Exception as e:
        return build_error_response(
            "CONVERSATIONS_RETRIEVAL_FAILED",
            f"Failed to retrieve conversations: {str(e)}",
            500
        )

# Retrieve a single conversation by its string id
def get_conversation_by_id(convo_id: str):
    """
    Retrieve a single conversation by its ID.

    Args:
        convo_id (str): String ID of the conversation (MongoDB ObjectId).

    Returns:
        dict | None: The serialized conversation if found, else None.
    
    Raises:
        JSONResponse: Error response if retrieval fails or ID is invalid.
    """
    try:
        if not convo_id:
            return build_error_response(
                "INVALID_INPUT",
                "Conversation ID is required",
                400
            )
        
        if not ObjectId.is_valid(convo_id):
            return build_error_response(
                "INVALID_CONVERSATION_ID",
                "Invalid conversation ID format",
                400
            )
        
        convo = conversation_collection.find_one({"_id": ObjectId(convo_id)})
        if convo:
            return serialize_mongo_document(convo)
        
        return build_error_response(
            "CONVERSATION_NOT_FOUND",
            "Conversation not found",
            404
        )
    except Exception as e:
        return build_error_response(
            "CONVERSATION_RETRIEVAL_FAILED",
            f"Failed to retrieve conversation: {str(e)}",
            500
        )

# Add a user or bot message to an existing conversation
async def add_message(convo_id: str, message: Message, response: str):
    """
    Add a user message and corresponding assistant response to a conversation.

    Args:
        convo_id (str): ID of the conversation.
        message (Message): Message object from the user.
        response (str): Assistant-generated response.

    Side Effects:
        - Updates the MongoDB conversation.
        - Adds corresponding vectors to Qdrant for semantic search and history tracking.

    Returns:
        dict: Success message or error response.
    
    Raises:
        JSONResponse: Error response if message addition fails.
    """
    try:
        if not convo_id or not message or not response:
            return build_error_response(
                "INVALID_INPUT",
                "Conversation ID, message, and response are required",
                400
            )
        
        if not ObjectId.is_valid(convo_id):
            return build_error_response(
                "INVALID_CONVERSATION_ID",
                "Invalid conversation ID format",
                400
            )
        
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
        update_result = conversation_collection.update_one(
            {"_id": ObjectId(convo_id)},
            {"$push": {"messages": {"$each": [msg, bot_reply]}}}
        )
        
        if update_result.matched_count == 0:
            return build_error_response(
                "CONVERSATION_NOT_FOUND",
                "Conversation not found",
                404
            )
        
        # Add message to Qdrant for history tracking
        # Lookup conversation
        convo = conversation_collection.find_one({"_id": ObjectId(convo_id)})
        if not convo:
            return build_error_response(
                "CONVERSATION_NOT_FOUND",
                "Conversation not found after update",
                404
            )
        
        # Extract user_id from the conversation document
        user_id = convo["user_id"]

        # Store the message and bot response vector in Qdrant for retrieval/history
        await add_message_vector(
            collection_name=user_id,
            conversation_id=convo_id,
            user_message=message.content,
            bot_response=response,
            timestamp=msg["timestamp"].isoformat(),
        )
        
        return {"message": "Message added successfully", "conversation_id": convo_id}
        
    except Exception as e:
        return build_error_response(
            "MESSAGE_ADDITION_FAILED",
            f"Failed to add message: {str(e)}",
            500
        )

"""Update the title of a conversation"""
def update_conversation_title(convo_id: str, new_title: str):
    """
    Update the title of a specific conversation.

    Args:
        convo_id (str): ID of the conversation to update.
        new_title (str): New title to assign.

    Returns:
        dict: The updated conversation document or error response.
    
    Raises:
        JSONResponse: Error response if update fails.
    """
    try:
        if not convo_id or not new_title:
            return build_error_response(
                "INVALID_INPUT",
                "Conversation ID and new title are required",
                400
            )
        
        if not ObjectId.is_valid(convo_id):
            return build_error_response(
                "INVALID_CONVERSATION_ID",
                "Invalid conversation ID format",
                400
            )
        
        result = conversation_collection.update_one(
            {"_id": ObjectId(convo_id)},
            {"$set": {"title": new_title}}
        )
        
        if result.matched_count == 0:
            return build_error_response(
                "CONVERSATION_NOT_FOUND",
                "Conversation not found",
                404
            )
        
        if result.modified_count == 0:
            return build_error_response(
                "TITLE_UPDATE_FAILED",
                "Title update failed - no changes made",
                400
            )
            
        # Return the updated conversation
        updated_convo = conversation_collection.find_one({"_id": ObjectId(convo_id)})
        return serialize_mongo_document(updated_convo)
        
    except Exception as e:
        return build_error_response(
            "TITLE_UPDATE_FAILED",
            f"Failed to update conversation title: {str(e)}",
            500
        )

async def delete_conversation_by_id(conversation_id: str, user_id: str) -> Dict:
    """
    Delete a conversation and its associated vectors in both MongoDB and Qdrant.

    Args:
        conversation_id (str): The ID of the conversation to delete.
        user_id (str): The ID of the user to ensure ownership.

    Returns:
        dict: A message indicating the result and the conversation ID.

    Raises:
        JSONResponse: Error response if deletion fails.
    """
    try:
        if not conversation_id or not user_id:
            return build_error_response(
                "INVALID_INPUT",
                "Conversation ID and user ID are required",
                400
            )
        
        if not ObjectId.is_valid(conversation_id):
            return build_error_response(
                "INVALID_CONVERSATION_ID",
                "Invalid conversation ID format",
                400
            )

        # Step 1: Delete from MongoDB
        result = conversation_collection.delete_one({
            "_id": ObjectId(conversation_id),
            "user_id": user_id
        })

        if result.deleted_count == 0:
            return build_error_response(
                "CONVERSATION_NOT_FOUND",
                "Conversation not found or already deleted",
                404
            )

        # Step 2: Delete from Qdrant
        try:
            await delete_conversation_vectors(collection_name=user_id, conversation_id=conversation_id)
        except Exception as e:
            return build_error_response(
                "QDRANT_DELETION_FAILED",
                f"Deleted in MongoDB but failed in Qdrant: {str(e)}",
                500
            )

        return {"message": "Conversation deleted from MongoDB and Qdrant", "conversation_id": conversation_id}
        
    except Exception as e:
        return build_error_response(
            "CONVERSATION_DELETION_FAILED",
            f"Failed to delete conversation: {str(e)}",
            500
        )

def serialize_user(user):
    """
    Convert a MongoDB user document into a serializable API format.

    Args:
        user (dict): Raw MongoDB user document.

    Returns:
        dict | None: Cleaned user data including `id`, `fullName`, `email`, and `phone`.
    """
    if not user:
        return None
    return {
        "id": str(user.get("_id", "")),  # ensures string
        "fullName": user.get("fullName", ""),
        "email": user.get("email", ""),
        "phone": user.get("phone", "")
    }


def register_user(user_data: UserCreate):
    """
    Register a new user after validating uniqueness and hashing the password.

    Args:
        user_data (UserCreate): The user registration payload.

    Returns:
        dict: Serialized user object for API response or error response.

    Raises:
        JSONResponse: Error response if registration fails.
    """
    try:
        print("Registering user function:", user_data)
        
        if not user_data.email or not user_data.password:
            return build_error_response(
                "INVALID_INPUT",
                "Email and password are required",
                400
            )
        
        existing_user = user_collection.find_one({"email": user_data.email})
        if existing_user:
            return build_error_response(
                "USER_ALREADY_EXISTS",
                "User with this email already exists",
                409
            )

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
        
    except Exception as e:
        return build_error_response(
            "USER_REGISTRATION_FAILED",
            f"Failed to register user: {str(e)}",
            500
        )


def login_user(credentials: UserLogin):
    """
    Authenticate a user using email and password.

    Args:
        credentials (UserLogin): Login request containing email and password.

    Returns:
        dict: Serialized user if authentication is successful, else error response.
    
    Raises:
        JSONResponse: Error response if login fails.
    """
    try:
        if not credentials.email or not credentials.password:
            return build_error_response(
                "INVALID_INPUT",
                "Email and password are required",
                400
            )
        
        user = user_collection.find_one({"email": credentials.email})
        if not user:
            return build_error_response(
                "INVALID_CREDENTIALS",
                "Invalid email or password",
                401
            )
        
        if not bcrypt.checkpw(credentials.password.encode("utf-8"), user["password"].encode("utf-8")):
            return build_error_response(
                "INVALID_CREDENTIALS",
                "Invalid email or password",
                401
            )
        
        return serialize_user(user)
        
    except Exception as e:
        return build_error_response(
            "LOGIN_FAILED",
            f"Login failed: {str(e)}",
            500
        )
