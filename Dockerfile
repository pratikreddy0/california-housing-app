# Use the official Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt into the container at /app
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Copy local directories into the container (adjust the local paths as needed)
COPY ./data /app/data
COPY ./model /app/model
COPY ./model_metrics /app/model_metrics
COPY ./logs /app/logs

# Copy the entire project directory into the container
COPY . .

# Expose the port Streamlit uses
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
