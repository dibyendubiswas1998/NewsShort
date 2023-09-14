FROM python:3.11.4-slim-buster

RUN apt update -y && apt install awscli -y
RUN apt-get install -y git
WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt


# Expose port 8000
EXPOSE 8080

# Run the Flask app with the updated command to bind to 0.0.0.0
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8080"]

