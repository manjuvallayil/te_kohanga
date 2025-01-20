FROM python:3.8

# Copy project files
COPY . /app

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 5000

# Run the Flask app
CMD ["gunicorn", "--workers=4", "--bind", "0.0.0.0:5000", "app:app"]