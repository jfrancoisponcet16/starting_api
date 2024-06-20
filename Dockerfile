# Use an official NVIDIA CUDA base image
FROM nvidia/cuda:11.1-base-ubuntu20.04

# Set the working directory in the Docker image
WORKDIR /app

# Copy the requirements.txt file into the Docker image
COPY requirements.txt ./

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx

# Upgrade pip
RUN pip3 install --upgrade pip

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Copy the rest of the code into the Docker image
COPY . .

# Expose port 8000 for the API
EXPOSE 8000

# Define the command to run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]