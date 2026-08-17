FROM python:3.11-slim

# Install Tesseract and poppler
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy shared package first (for layer caching)
COPY shared/ /app/shared/

# Copy service code
COPY services/tesseract/ /app/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]