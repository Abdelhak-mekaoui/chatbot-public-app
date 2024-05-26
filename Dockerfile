# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

# Set the environment variable for MongoDB URI
ENV MONGO_URI="mongodb://root:password@localhost:27017"
ENV MONGO_DB="chat-docs"
ENV MONGO_COLLECTION="docs"

# Set the environment variable to disable uvloop
ENV UVLOOP_DISABLED="1"


# Expose the port number the app runs on
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
