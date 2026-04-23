FROM python:3.10

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Run FastAPI using uvicorn
CMD uvicorn app:app --host $HOST --port $PORT