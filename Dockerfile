# Use the official PyTorch base image that supports CUDA (if available) for GPU acceleration.
# To build a CPU-only image, you might consider using a different base image such as python:3.11-slim.
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set the working directory inside the container.
WORKDIR /app

# Copy the requirements file and install dependencies.
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code.
COPY . .

# (Optional) Expose ports or add environment variables if needed.

# Run the training script.
CMD ["python", "src/train.py"]

# docker build -t frillsformer .
# docker run frillsformer
