FROM python:3.11-slim

# Prevent Python from writing .pyc files and make output unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system dependencies, Chromium and Chromedriver
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl unzip gnupg ca-certificates \
    chromium chromium-driver \
    libglib2.0-0 libnss3 libgconf-2-4 libfontconfig1 libxss1 \
    libappindicator3-1 libasound2 libatk-bridge2.0-0 libatk1.0-0 \
    libcups2 libdbus-1-3 libgdk-pixbuf2.0-0 libnspr4 libx11-xcb1 \
    libxcomposite1 libxdamage1 libxrandr2 libgbm1 xdg-utils \
    && rm -rf /var/lib/apt/lists/*

# Set Chromium binary location for Selenium
ENV CHROME_BIN=/usr/bin/chromium

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN crawl4ai-setup

# Copy app code
COPY . .

# Expose app port
EXPOSE 8080

# Run the Flask app via Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "1200", "app:app"]
