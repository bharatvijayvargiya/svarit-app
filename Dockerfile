# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Expose port (optional, for documentation)
EXPOSE 10000

# Run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
