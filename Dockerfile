FROM python:3.12

WORKDIR /usr/src/personalised_nudges

COPY ./app ./app
COPY ./vizualization ./vizualization

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

#CMD ["/bin/bash", "-c", "python - m app.bot.main"]