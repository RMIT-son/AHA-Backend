import os
from dotenv import load_dotenv

load_dotenv()

APP_CONFIG = {
    "app": {
        "name": "AHA-Capstone",
        "server_url": os.getenv("BASE_API_URL"),
        "client_url": os.getenv("CLIENT_URL"),
    },
    "port": int(os.getenv("PORT", 8000)),
    "database": {
        "url": os.getenv("MONGO_DB_URL"),
    },
    "jwt": {
        "secret": os.getenv("JWT_SECRET"),
        "token_life": "7d",  # You can make this configurable via env too
    }
}
