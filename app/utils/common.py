from fastapi.responses import JSONResponse
from datetime import datetime

def create_signature_with_doc(base_cls, docstring: str):
    """
    Dynamically create a subclass of the given base class with a custom docstring.

    Useful for modifying or documenting classes at runtime (e.g., in API schemas or DSLs).

    Args:
        base_cls (type): The base class to extend.
        docstring (str): The new docstring to apply to the dynamically created class.

    Returns:
        type: A new subclass of `base_cls` with the specified docstring.
    """
    return type(base_cls.__name__, (base_cls,), {"__doc__": docstring})

def build_error_response(code: str, message: str, status: int) -> JSONResponse:
    """
    Construct a standardized JSON error response for API endpoints.

    This helper function formats error responses according to a consistent schema
    containing an error code, message, HTTP status, and a UTC timestamp.

    Args:
        code (str): A short error code identifier (e.g., "RESOURCE_NOT_FOUND").
        message (str): A human-readable error message.
        status (int): HTTP status code (e.g., 400, 404, 500).

    Returns:
        JSONResponse: A FastAPI JSONResponse object containing the formatted error.
    
    Example:
        return build_error_response("INVALID_ID", "The provided ID is not valid.", 400)
    """
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "code": code,
                "message": message,
                "status": status,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
    )


# Helper function to convert MongoDB document (_id) into a serializable dictionary
def serialize_mongo_document(doc):
    """
    Convert a MongoDB document into a serializable dictionary for API responses.

    Replaces the `_id` field with a string `id` for frontend compatibility.

    Args:
        doc (dict): A MongoDB document.

    Returns:
        dict | None: A sanitized and serializable dictionary, or None if the input is falsy.
    """
    if not doc:
        return None
    
    doc = doc.copy()
    if "_id" in doc:
        doc["id"] = str(doc["_id"])  # Replace MongoDB's _id with stringified id
        del doc["_id"]
    return doc