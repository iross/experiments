FROM python:3.8-slim
RUN pip install --upgrade pip
WORKDIR /app/
COPY requirements.txt /app/
RUN pip install -r requirements.txt
COPY search.py/ /app/
RUN ["python", "-c", "import search; _ = search.MPNetEmbedder()"] # instantiate to download models, vocabs, etc
CMD ["python", "search.py"]
