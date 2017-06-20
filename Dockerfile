FROM tensorflow/tensorflow:latest-py3

MAINTAINER Georg Wiese <georgwiese@gmail.com>

RUN apt-get update
RUN apt-get -y install git htop

ADD requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt

ADD biomedical_qa /biomedical_qa
ADD start_server.sh /biomedical_qa
ADD final_model /model

WORKDIR /biomedical_qa
ENV PYTHONPATH /

EXPOSE 5000

CMD ./start_server.sh single
