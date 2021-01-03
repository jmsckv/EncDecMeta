#!/bin/bash

. ./set_env_vars.sh

env > .configured_env_vars_docker

docker build . -t $IMAGENAME


docker run -dit --name $CONTAINERNAME  \
            --restart always \
            -p $PORT1:9000 -p $PORT2:6006 -p $PORT3:8265 \
            -v $DATAPATH:/data \
            -v $RESULTSPATH:/results \
            -v $CODEPATH:/code \
           $IMAGENAME\
           jupyter lab --port=9000 --no-browser --allow-root
           # default password is ASHA2020 > change its hash in jupyter_notebook_config.py 


# explananation of build args
# --restart always > restart policy which restarts a container if it non-manually stops or if the Docker daemon restarts
# -u "$(id -u):$(id -g)" > run with same privileges as on host,
# e.g. this should avoid creating files with root permission while not having root rights on host
# the problem I face when using this flag is to pip install packages "[Errno 13] Permission denied: '/.local'"


           
          
