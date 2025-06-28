from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from api.supervisor import router, lifespan
from api.routes import conversation

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
