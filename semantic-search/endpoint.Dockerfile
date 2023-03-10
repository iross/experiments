FROM python:3.8-slim
RUN pip install --upgrade pip
WORKDIR /app/
COPY requirements.txt /app/
RUN pip install -r requirements.txt
WORKDIR /app/src/
COPY search.py /app/src/
RUN ["python", "-c", "import search; _ = search.MPNetEmbedder()"] # instantiate to download models, vocabs, etc

WORKDIR /app/
COPY endpoint_requirements.txt /app/
RUN pip install -r endpoint_requirements.txt
WORKDIR /app/src/
COPY src/ /app/src
COPY wsgi.py /app/src/
WORKDIR /app/src
CMD ["gunicorn", "-b", "0.0.0.0:5000", "wsgi:app"]j
