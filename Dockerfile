# base image
FROM python:3.10-slim

# set working directory
WORKDIR /app

# copy requirements
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy application code
COPY . .

# set pythonpath
ENV PYTHONPATH=/app

# expose streamlit port
EXPOSE 8501

# run streamlit
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
