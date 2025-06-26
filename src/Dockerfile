# Use the official Python image as a base image
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Install OS-level dependencies needed by some packages (e.g., Pillow, torch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file initially for better caching of dependency installation
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# (Optional) Expose a port if you plan to use a web server or monitoring tool
# EXPOSE 8888

# Set the default command to run your training script.
# You can change this to run other scripts (e.g., src/predict.py for inference).
CMD ["python", "src/train.py"]
