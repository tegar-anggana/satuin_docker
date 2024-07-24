# Base image with common dependencies
FROM python:3.10-slim AS base

# Set up environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install common dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    pip install --no-cache-dir Flask==2.2.5 torch==1.12.1

# Stage for App A specific dependencies
FROM base AS a-dependencies

# Create and set the working directory
WORKDIR /app

# Copy only the requirements file first to leverage Docker caching
COPY ./a-backend-main/requirements.txt .

# Install App A specific dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code
COPY ./a-backend-main .

# Expose the port your application will run on
EXPOSE 8881

# Specify the command to run on container start
CMD ["python", "app.py"]

# Stage for App B specific dependencies
FROM base AS b-dependencies

# Create and set the working directory
WORKDIR /app

# Copy only the requirements file first to leverage Docker caching
COPY ./skripsi-deployment-main/requirements.txt .

# Install App B specific dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code
COPY ./skripsi-deployment-main .

# Expose the port your application will run on
EXPOSE 8882

# Specify the command to run on container start
CMD ["python", "app.py"]
