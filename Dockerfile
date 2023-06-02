FROM python:3.9.16

WORKDIR /Oracle

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .

RUN pip install -r requirements.txt
