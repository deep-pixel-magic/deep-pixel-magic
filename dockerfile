# Use the python base image.
FROM python:latest

# Specify the working directory.
WORKDIR /app

# Set up virtual environment.
RUN python -m venv .venv

# Copy requirements.txt to the working directory.
COPY requirements.txt .

# Update to the latest version of pip.
RUN pip3 install --upgrade pip

# Install the dependencies.
RUN pip3 install -r requirements.txt

