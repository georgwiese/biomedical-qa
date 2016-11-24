FROM tensorflow/tensorflow

MAINTAINER Georg Wiese <georgwiese@gmail.com>

RUN apt-get update
RUN apt-get -y install ipython3 ipython3-notebook python3-pip git libblas-dev liblapack-dev gfortran
RUN pip3 install ipykernel
RUN python3 -m ipykernel install --user

ADD requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt
RUN pip3 install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0-cp34-cp34m-linux_x86_64.whl

WORKDIR /biomedical-qa
ENV PYTHONPATH /biomedical-qa
CMD jupyter notebook
