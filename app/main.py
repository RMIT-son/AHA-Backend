from fastapi import FastAPI
from app.api.routes import conversation
from contextlib import asynccontextmanager
from app.services.manage_models.model_manager import model_manager
from fastapi.middleware.cors import CORSMiddleware
import asyncio

@asynccontextmanager
async def lifespan(app):
    """Application lifespan manager for model loading and cleanup."""
    try:
        # Load models immediately (fast)
        model_manager.load_models()

        # Warm up models asynchronously in background
        asyncio.create_task(model_manager.warmup_models())

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

app.include_router(conversation.router)