#FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04
FROM nvidia/cuda:11.6.0-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y python3-pip

RUN mkdir /model

WORKDIR /model

ADD requirements.txt /model

RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

ADD . /model/
