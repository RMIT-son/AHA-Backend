import bcrypt
from typing import Dict
from bson import ObjectId
from datetime import datetime
from fastapi import HTTPException
from database.schemas import UserCreate, UserLogin, Message
from .mongo_client import conversation_collection, user_collection
from database.qdrant_client import add_message_vector, delete_conversation_vectors

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

def serialize_file_data(file_data):
    """Convert file data to MongoDB-friendly format"""
    if not file_data:
        return None
    
    return {
        "name": file_data.name,
        "type": file_data.type,
        "size": file_data.size,
        "data": file_data.data  # Store the full base64 data URL
    }

def serialize_message_for_db(message: Message):
    """Convert Message object to MongoDB-friendly format"""
    msg_dict = {
        "content": message.content,
        "timestamp": message.timestamp or datetime.utcnow()
    }
    
    # Handle legacy image field
    if message.image:
        msg_dict["image"] = message.image
    
    # Handle new files field
    if message.files:
        msg_dict["files"] = [serialize_file_data(file_data) for file_data in message.files]
    
    return msg_dict

def extract_content_for_vector(message: Message) -> str:
    """Extract meaningful content from message for vector storage"""
    content_parts = []
    
    # Add text content
    if message.content and message.content.strip():
        content_parts.append(message.content.strip())
    
    # Add file descriptions
    if message.files:
        for file_data in message.files:
            if file_data.type.startswith('image/'):
                content_parts.append(f"[Image: {file_data.name}]")
            elif file_data.type == 'application/pdf':
                content_parts.append(f"[PDF: {file_data.name}]")
            else:
                content_parts.append(f"[File: {file_data.name}]")
    
    # Handle legacy image field
    if message.image and not any('Image:' in part for part in content_parts):
        content_parts.append("[Image attachment]")
    
    return " ".join(content_parts) if content_parts else "[No content]"

# Create a new conversation document in the database
def create_conversation(user_id: str, title: str):
    """Create a new conversation for a user"""
    try:
        convo = {
            "title": title,
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "messages": []
        }
        result = conversation_collection.insert_one(convo)

        # Add the inserted ObjectId as a string id for frontend compatibility
        convo["id"] = str(result.inserted_id)
        
        print(f"Created conversation: {convo['id']} for user: {user_id}")
        return convo
    except Exception as e:
        print(f"Error creating conversation: {e}")
        raise

# Retrieve all conversation documents and serialize ObjectId to id
def get_all_conversations(user_id: str):
    """Get all conversations for a specific user"""
    try:
        # Only get conversations belonging to this user, sorted by creation date (newest first)
        conversations = list(
            conversation_collection.find({"user_id": user_id})
            .sort("created_at", -1)
        )
        
        for convo in conversations:
            if "_id" in convo:
                convo["id"] = str(convo["_id"])
                del convo["_id"]
        
        print(f"Retrieved {len(conversations)} conversations for user: {user_id}")
        return conversations
    except Exception as e:
        print(f"Error getting conversations for user {user_id}: {e}")
        return []

# Retrieve a single conversation by its string id
def get_conversation_by_id(convo_id: str):
    """Get a specific conversation by ID"""
    try:
        convo = conversation_collection.find_one({"_id": ObjectId(convo_id)})
        if convo:
            serialized = serialize_mongo_document(convo)
            print(f"Retrieved conversation: {convo_id}")
            return serialized
        print(f"Conversation not found: {convo_id}")
        return None
    except Exception as e:
        # If ObjectId is invalid (e.g. wrong format), catch and log
        print(f"Error finding conversation {convo_id}: {e}")
        return None

# Add a user or bot message to an existing conversation
async def add_message(convo_id: str, message: Message, response: str):
    """Add user message and bot response to conversation"""
    try:
        # Serialize user message with files
        user_msg = serialize_message_for_db(message)
        user_msg["sender"] = "user"
        
        print(f"Adding user message to conversation {convo_id}")
        print(f"Message content: {message.content}")
        print(f"Files: {len(message.files) if message.files else 0}")
        
        # Create bot reply
        bot_reply = {
            "sender": "assistant",
            "content": response,
            "timestamp": datetime.utcnow()
        }
        
        # Push both user message and bot reply into the conversation
        result = conversation_collection.update_one(
            {"_id": ObjectId(convo_id)},
            {"$push": {"messages": {"$each": [user_msg, bot_reply]}}}
        )
        
        if result.modified_count == 0:
            print(f"Warning: No documents modified when adding message to {convo_id}")
            return
        
        # Add message to Qdrant for history tracking
        try:
            # Lookup conversation to get user_id
            convo = conversation_collection.find_one({"_id": ObjectId(convo_id)})
            if not convo:
                print(f"Warning: Conversation {convo_id} not found for vector storage")
                return
            
            user_id = convo["user_id"]
            
            # Extract content for vector storage
            user_content = extract_content_for_vector(message)
            
            print(f"Storing vector for conversation {convo_id}, user {user_id}")
            print(f"Vector content: {user_content}")
            
            # Store the message and bot response vector in Qdrant for retrieval/history
            await add_message_vector(
                collection_name=user_id,
                conversation_id=convo_id,
                user_message=user_content,
                bot_response=response,
                timestamp=user_msg["timestamp"].isoformat(),
            )
            
        except Exception as vector_error:
            print(f"Error storing message vector: {vector_error}")
            # Don't fail the whole operation if vector storage fails
            
    except Exception as e:
        print(f"Error adding message to conversation {convo_id}: {e}")
        raise

def update_conversation_title(convo_id: str, new_title: str):
    """Update the title of a conversation"""
    try:
        result = conversation_collection.update_one(
            {"_id": ObjectId(convo_id)},
            {"$set": {"title": new_title}}
        )
        
        if result.modified_count == 0:
            print(f"No conversation found or updated for ID: {convo_id}")
            return None
            
        # Return the updated conversation
        updated_convo = conversation_collection.find_one({"_id": ObjectId(convo_id)})
        print(f"Updated conversation title: {convo_id} -> {new_title}")
        return serialize_mongo_document(updated_convo)
        
    except Exception as e:
        print(f"Error updating conversation title: {e}")
        return None

async def delete_conversation_by_id(conversation_id: str, user_id: str) -> Dict:
    """Delete a conversation from both MongoDB and Qdrant"""
    if not ObjectId.is_valid(conversation_id):
        raise HTTPException(status_code=400, detail="Invalid conversation ID")

    try:
        # Step 1: Delete from MongoDB
        result = conversation_collection.delete_one({
            "_id": ObjectId(conversation_id),
            "user_id": user_id
        })

        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Conversation not found or already deleted")

        print(f"Deleted conversation {conversation_id} from MongoDB")

        # Step 2: Delete from Qdrant
        try:
            await delete_conversation_vectors(collection_name=user_id, conversation_id=conversation_id)
            print(f"Deleted conversation {conversation_id} vectors from Qdrant")
        except Exception as e:
            print(f"Warning: Failed to delete vectors from Qdrant: {e}")
            # Don't fail the whole operation if vector deletion fails
            raise HTTPException(status_code=500, detail=f"Deleted in MongoDB but failed in Qdrant: {str(e)}")

        return {"message": "Conversation deleted from MongoDB and Qdrant", "conversation_id": conversation_id}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting conversation: {str(e)}")

def serialize_user(user):
    """Convert user document to API-friendly format"""
    if not user:
        return None
    return {
        "id": str(user.get("_id", "")),  # ensures string
        "fullName": user.get("fullName", ""),
        "email": user.get("email", ""),
        "phone": user.get("phone", "")
    }

def register_user(user_data: UserCreate):
    """Register a new user"""
    try:
        print(f"Registering user: {user_data.email}")
        
        existing_user = user_collection.find_one({"email": user_data.email})
        if existing_user:
            raise ValueError("User already exists")

        hashed_pw = bcrypt.hashpw(user_data.password.encode("utf-8"), bcrypt.gensalt())

        new_user = {
            "fullName": user_data.fullName,
            "email": user_data.email,
            "password": hashed_pw.decode("utf-8"),  # Store as string
            "phone": user_data.phone,
            "created_at": datetime.utcnow()
        }
        
        result = user_collection.insert_one(new_user)
        new_user["_id"] = result.inserted_id
        
        print(f"User registered successfully with ID: {result.inserted_id}")
        return serialize_user(new_user)
        
    except ValueError:
        raise
    except Exception as e:
        print(f"Error registering user: {e}")
        raise

def login_user(credentials: UserLogin):
    """Authenticate user login"""
    try:
        user = user_collection.find_one({"email": credentials.email})
        if user and bcrypt.checkpw(credentials.password.encode("utf-8"), user["password"].encode("utf-8")):
            print(f"User logged in successfully: {credentials.email}")
            return serialize_user(user)
        
        print(f"Login failed for: {credentials.email}")
        return None
        
    except Exception as e:
        print(f"Error during login: {e}")
        return None