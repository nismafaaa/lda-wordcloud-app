# Base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the rest of the application
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Command to run the application
CMD ["python", "app.py"]
