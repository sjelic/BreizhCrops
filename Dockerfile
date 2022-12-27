FROM pytorch/pytorch:latest
WORKDIR /workdir
COPY . /workdir
ENV PYTHONPATH /workdir
RUN pip install -r /workdir/requirements.txt
RUN apt-get update
RUN apt-get install wget -y