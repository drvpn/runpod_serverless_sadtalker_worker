# Use the official Python 3.8 image from the Docker Hub
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Install dependencies and essential tools
RUN apt-get update && \
    apt-get install -y git ffmpeg && \
    apt-get clean

# Clone the SadTalker repository
RUN git clone https://github.com/OpenTalker/SadTalker.git /app/SadTalker

# Change to the SadTalker directory
WORKDIR /app/SadTalker

COPY app/ /app/SadTalker

# Install PyTorch with CUDA support and other dependencies
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 boto3 runpod==1.6.0 --extra-index-url https://download.pytorch.org/whl/cu113 && \
    pip install -r requirements.txt

# Set the entrypoint
CMD ["python", "-u", "/app/SadTalker/handler.py"]
