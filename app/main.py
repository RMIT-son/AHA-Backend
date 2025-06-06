# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import supervisor
from api.routes import conversation

app = FastAPI()

# === CORS Configuration for Local Frontend Access ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",  # Add other ports if needed
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(supervisor.router)
app.include_router(conversation.router)