# Use a slim version of Python to keep the image small
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (needed for some ML libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies with no-cache to save space
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port Flask/Gunicorn will run on
EXPOSE 8080

# Use Gunicorn as the production web server
# Matches your application.py naming convention
# CMD ["gunicorn", "--bind", "0.0.0.0:8080", "application:application"]
CMD ["python3", "application.py"]