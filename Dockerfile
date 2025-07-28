# Use official Python image
FROM python:3.10-slim

# Install build dependencies
RUN apt-get update && apt-get install -y gcc g++ curl

# Set working directory
WORKDIR /app

# Copy all files into the container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download spaCy model at build time
RUN python -m spacy download en_core_web_sm

# Expose FastAPI port
EXPOSE 10000

# Start the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
