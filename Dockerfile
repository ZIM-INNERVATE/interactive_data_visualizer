FROM python:3.10-slim

WORKDIR /work

COPY requirements.txt /

RUN pip install -r /requirements.txt

COPY ./ ./

EXPOSE 8085

ENTRYPOINT ["gunicorn", "--config", "gunicorn_config.py", "index:server"]
