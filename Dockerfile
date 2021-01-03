# 4 Build stages would make sense
# 1. Pytorch Base Image, Develop and Debug from within IDE / command line
# 2. Additional Python Libraries to discuss results: Jupyterlab, Maplotlib, ...
# 3. Test Libraries
# 4. Libraries to package and pulish to PyPi

FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime AS base_image


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
    rsync\
    ssh


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

EXPOSE 8888
EXPOSE 6006
EXPOSE 8265


# we use caching: add requirements > install requirements

# Setting up JupyterLab

ADD jupyterlab_requirements.txt . 
RUN pip install -r jupyterlab_requirements.txt
COPY jupyter_notebook_config.py /root/.jupyter/
COPY .bashrc /root/

# Install EncDecMeta in editable mode, takes longer
COPY . .
RUN pip install -e .

