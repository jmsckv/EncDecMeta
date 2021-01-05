FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
# FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime AS base_image
# note: the torch image comes with conda pre-installed but we don't make any use of it

RUN apt-get update -qq && apt-get install -y -qq \
    #wget \
    nano \
    curl \
    #screen \
    #vim \
    git \
    tmux \
    #cmake \
    tree\
    rsync
    #ssh\

SHELL ["/bin/bash", "-c"]

ENV DATAPATH=/data
ENV RAW_DATAPATH=/data/raw
ENV PROC_DATAPATH=/data/proc
ENV RESULTSPATH=/results
ENV CODEPATH=/code
RUN mkdir -p $CODEPATH
RUN mkdir -p $PROC_DATAPATH
RUN mkdir -p $RAW_DATAPATH
RUN mkdir -p $RESULTSPATH

WORKDIR $CODEPATH

# Setting up JupyterLab
# we use caching: add requirements > install requirements
ADD jupyterlab_requirements.txt . 
RUN pip install -r jupyterlab_requirements.txt
COPY jupyter_notebook_config.py /root/.jupyter/
COPY .bashrc /root/

# install package in editable mode
#ADD . .
#RUN pip install -e .