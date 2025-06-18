FROM python:3.11-slim

# Set working directory
WORKDIR /app

# ----> UPDATE THIS LINE <----
# Install system build dependencies required by some Python packages
# Add zlib1g-dev to the list of packages
RUN apt-get update && apt-get install -y build-essential zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
# -----------------------------

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else
COPY . .

# Expose the port (optional, for documentation)
EXPOSE 8080

# Start using the dynamic PORT from env
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]