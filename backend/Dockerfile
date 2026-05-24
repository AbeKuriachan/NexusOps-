FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (leverages Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
