# FROM python:3.11-alpine
FROM python:3.10-slim

# Set up environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install ffmpeg and other dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg

# Create and set the working directory
WORKDIR /app

# Copy only the requirements file first to leverage Docker caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code
COPY . .

# Expose the port your application will run on
EXPOSE 8881

# Specify the command to run on container start
CMD ["python", "app.py"]