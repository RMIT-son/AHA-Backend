import os
from fastapi import FastAPI
from api.supervisor import router, lifespan
from api.routes import conversation
from contextlib import asynccontextmanager
from services.model_manager import model_manager
from fastapi.middleware.cors import CORSMiddleware
from app.middleware import LoggingMiddleware

@asynccontextmanager
async def lifespan(app):
    """Application lifespan manager for model loading and cleanup."""
    try:
        # Load and warm up models on startup
        model_manager.load_models()
        await model_manager.warmup_models()
        print("Application startup completed successfully!")
        yield
        
    except Exception as e:
        print(f"Error during startup: {e}")
        raise
    finally:
        # Clean up models on shutdown
        model_manager.cleanup_models()
        print("Application shutdown completed successfully!")

app = FastAPI(lifespan=lifespan)

# === CORS Configuration for Local Frontend Access ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",

    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Logging Middleware ===
app.add_middleware(LoggingMiddleware)

app.include_router(router)
app.include_router(conversation.router)

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI on Cloud Run!"}

if __name__ == "__main__":
    import uvicorn
    import os
    import traceback

    try:
        port = int(os.environ.get("PORT", 8080))
        print(f"🚀 Starting FastAPI on port {port}")
        uvicorn.run("app.main:app", host="0.0.0.0", port=port)
    except Exception as e:
        print("❌ Failed to start FastAPI")
        traceback.print_exc()
