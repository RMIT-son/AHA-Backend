from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from database.schemas import UserCreate, UserLogin, UserResponse

from database.queries import register_user, login_user

# Create router with prefix and tag
router = APIRouter(prefix="/api/auth", tags=["Auth"])

# Endpoint to register a new user
@router.post("/register", response_model=UserResponse)
def register(user: UserCreate):
    """
    Register a new user in the system.

    Args:
        user (UserCreate): The user registration data including email, password, and any other required fields.

    Raises:
        HTTPException: If the registration data is invalid (e.g., email already exists).

    Returns:
        UserResponse: The registered user data including user ID and other non-sensitive information.
    """
    try:
        result = register_user(user)
        print("Serialized user result:", result)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print("Unexpected error:", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Endpoint to login a user
@router.post("/login", response_model=UserResponse)
def login(user: UserLogin):
    """
    Authenticate an existing user and return user information.

    Args:
        user (UserLogin): The user login credentials, typically email and password.

    Raises:
        HTTPException: If authentication fails due to invalid credentials.

    Returns:
        UserResponse: The authenticated user data including user ID and profile details.
    """
    try:
        result = login_user(user)
        if not result:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return result
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
