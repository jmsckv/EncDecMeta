#!/bin/bash

docker run -dit --name thesis_image_d071503 -p 9503:9000 -p 6503:6006 -p 8265:8265 --runtime=nvidia \
           -v $DATAPATH:/data \
           -v $RESULTSPATH:/results \
           -v $PYTHONPATH:/code \
           thesis_image jupyter lab --port=9000 --no-browser --allow-root



           
          
