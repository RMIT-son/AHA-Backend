from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import supervisor
from api.routes import conversation

app = FastAPI()

# === CORS Configuration for Local Frontend Access ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(supervisor.router)
app.include_router(conversation.router)