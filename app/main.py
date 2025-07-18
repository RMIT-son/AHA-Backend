from fastapi import FastAPI
from app.api.routes import conversation
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from app.services.manage_models.model_manager import model_manager

@asynccontextmanager
async def lifespan(app):
    """
    Application lifespan manager for model initialization and cleanup.

    This function is registered with FastAPI's `lifespan` parameter to handle:
    - Loading required models at startup.
    - Warming up models asynchronously in the background.
    - Cleaning up models on application shutdown.

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None: Control is yielded back to FastAPI once startup is complete.

    Raises:
        Exception: If any error occurs during model loading or warmup, it is printed and re-raised.
    """
    try:
        # Load models immediately (fast)
        model_manager.load_models()

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(conversation.router)