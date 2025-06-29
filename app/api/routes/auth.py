from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from database.schemas import UserCreate, UserLogin, UserResponse

from database.queries import register_user, login_user

# Create router with prefix and tag
router = APIRouter(prefix="/api/auth", tags=["Auth"])

# Endpoint to register a new user
@router.post("/register", response_model=UserResponse)
def register(user: UserCreate):
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
    try:
        result = login_user(user)
        if not result:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return result
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)