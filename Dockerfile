# 2 stage build
# 1. Pytorch Base Image, Develop and Debug from within IDE / command line
# 2. Additional Python Libraries to discuss results: Jupyterlab, Pandas

FROM pytorch/pytorch AS base_image

SHELL ["/bin/bash", "-c"]

WORKDIR /work
ENV DATAPATH=/work/data
ENV RESULTSPATH=/work/results
ENV CODEPATH=/work/code
RUN mkdir -p $CODEPATH
RUN mkdir -p $DATAPATH
RUN mkdir -p $RESULTSPATH

COPY . $CODEPATH
RUN cd $CODEPATH && pip install -e .


##################################

FROM base_image AS extended_image

RUN cd ${CODEPATH} && pip install -e .[extended]
RUN jupyter nbextension enable --py widgetsnbextension

COPY jupyter_notebook_config.py /root/.jupyter/
COPY .bashrc /root/

EXPOSE 8888
EXPOSE 6006

#### TODO:image for testing


# [Optional] Allow the vscode user to pip install globally w/o sudo
#ENV PIP_TARGET=/usr/local/pip-global
#ENV PYTHONPATH=${PIP_TARGET}:${PYTHONPATH}
#ENV PATH=${PIP_TARGET}/bin:${PATH}
#RUN mkdir -p ${PIP_TARGET} \
#    && chown vscode:root ${PIP_TARGET} \
#    && echo "if [ \"\$(stat -c '%U' ${PIP_TARGET})\" != \"vscode\" ]; then chown -R vscode:root ${PIP_TARGET}; fi" \
#        | tee -a /root/.bashrc /home/vscode/.bashrc /root/.zshrc >> /home/vscode/.zshrc 

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
# COPY requirements.txt /tmp/pip-tmp/
# RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
#    && rm -rf /tmp/pip-tmp

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update \
#     && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
