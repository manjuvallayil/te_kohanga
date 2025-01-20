# Use the official Python image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 5000

# Start the app
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
