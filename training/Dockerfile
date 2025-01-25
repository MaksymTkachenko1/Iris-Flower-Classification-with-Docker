# Stage 1: Builder
FROM python:3.10-slim AS builder

WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Copy only the installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copy the necessary files from the project
COPY data_process/data_download.py /app/data_process/
COPY settings.json /app/
COPY training/ /app/training/

RUN chmod +x /app/data_process/data_download.py
RUN python /app/data_process/data_download.py

# Define the volume for data
VOLUME /app/data

# Train the model
CMD ["python", "training/train.py"]