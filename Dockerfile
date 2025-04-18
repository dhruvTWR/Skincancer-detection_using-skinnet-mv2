# Use official Python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Streamlit specific configs to avoid CLI prompt
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ENABLECORS=false

# Expose the port Streamlit runs on
EXPOSE 7860

# Run the app
CMD ["streamlit", "run", "app.py"]
