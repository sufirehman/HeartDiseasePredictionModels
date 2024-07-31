# Use an official Python runtime as a parent image
FROM python:3.12.4

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install necessary system packages
RUN apt-get update && apt-get install -y build-essential

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Check if Streamlit is installed (for debugging)
RUN streamlit --version

# Expose port 8501 for Streamlit
EXPOSE 8501

# Run app.py when the container launches
CMD ["python", "-m", "streamlit", "run", "app.py"]
