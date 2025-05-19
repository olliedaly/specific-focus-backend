# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Create a non-root user and group
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy the requirements file into the container at /app
COPY requirements.txt requirements.txt

# Install any needed packages specified in requirements.txt
# Ensure build dependencies are installed and then removed if not needed at runtime
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container at /app
COPY backend_gemini_app.py .
COPY .env .env # Optional: if you want to include a default .env for Cloud Run, though env vars are better

# Change ownership of the app directory to the non-root user
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Make port available to the world outside this container (Cloud Run will set PORT env var)
EXPOSE 8080

# Define environment variable (Cloud Run will set this, but good for local testing if not set)
ENV PORT 8080

# Run the FastAPI app using Uvicorn when the container launches
# Your backend_gemini_app.py already handles the PORT environment variable.
# The "reload=True" in your __main__ is for development; remove for production image
# Uvicorn command will be: uvicorn backend_gemini_app:app --host 0.0.0.0 --port $PORT
CMD ["uvicorn", "backend_gemini_app:app", "--host", "0.0.0.0", "--port", "8080"]