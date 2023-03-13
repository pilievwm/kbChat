FROM python:latest

WORKDIR /app

COPY requirements.txt requirements.txt
RUN python -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install tiktoken && \
    pip install -r requirements.txt

COPY app.py .
COPY botAnswer.py .
COPY vectors.py .
COPY searchBot.py .

VOLUME /app/data

ENV PATH="/app/venv/bin:$PATH"

CMD ["python", "app.py"]
