FROM python:3.12-slim

WORKDIR /app

# RUN apt-get update && apt-get install -y build-essential

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

# Run on port 8001 to avoid conflict if running locally, 
# though inside docker container port mapping handles it.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
