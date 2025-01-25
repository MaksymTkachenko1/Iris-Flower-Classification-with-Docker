# Stage 1: Builder
FROM python:3.10-slim AS builder

WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

# Stage 2: Runtime
FROM python:3.10-slim

# Set environment variables
ARG model_name=iris_model.pth
ENV MODEL_NAME=${model_name}

WORKDIR /app

# Copy only the installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copy the necessary files into the container
COPY data/ /app/data/
COPY models/ /app/models/
COPY inference/ /app/inference/
COPY training/ /app/training/  
COPY settings.json /app/

# Use ENTRYPOINT for the main command and CMD for default parameters
ENTRYPOINT ["python", "/app/inference/run.py"]
CMD ["--settings_path", "/app/settings.json"]
