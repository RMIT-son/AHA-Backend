FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Pre-copy only requirements.txt for caching
COPY requirements.txt .

# Install Python dependencies and cache this layer
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app (this layer will change often)
COPY . .

# Expose port for Cloud Run
EXPOSE 8000

# Run FastAPI via uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
