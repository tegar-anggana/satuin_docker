# Use the ffmpeg image as the base image
FROM jrottenberg/ffmpeg:4.4-alpine AS ffmpeg

# Use a Python image to build the application
FROM python:3.10

# Set up environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install ffmpeg from the ffmpeg image
COPY --from=ffmpeg / /

# Create and set the working directory
WORKDIR /app

# Copy only the requirements file first to leverage Docker caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code
COPY . .

# Expose the port your application will run on
EXPOSE 8080

# Specify the command to run on container start
CMD ["python", "app.py"]
