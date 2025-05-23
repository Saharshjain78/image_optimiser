FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package
RUN pip install -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PORT=8501

# Create volume for model storage
VOLUME /app/models

# Streamlit runs on port 8501 by default, FastAPI on 8000
EXPOSE 8501 8000

# Run the application
CMD ["streamlit", "run", "streamlit_app.py"]

# For API mode, use: CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
