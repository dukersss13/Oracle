FROM python:3.9.16

RUN pip install --upgrade pip
RUN pip freeze > requirements.txt
RUN pip install -r requirements.txt
