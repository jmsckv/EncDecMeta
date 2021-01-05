#!/bin/bash

# we use this pattern to set env vars: if the var is unset, set the var to
# [ -z "$ENV" ] && export ENV=<some_path>


# Docker-related
[ -z "$IMAGENAME" ] && export IMAGENAME=$(id -u)_encdec_image
[ -z "$CONTAINERNAME" ] && export CONTAINERNAME=$(id -u)_encdec


# Directory structure
# note: env vars maintained inside Docker container, mapping to env vars on host system
# directory structure relative to dir in which set_env_vars.sh resides
# the assuemd folder structure is:
# /Project # $PROJECTPATH
# /Project/Results # $RESULTSPATH
# /Project/Data/  # $DATAPATH
# /Project/Data/raw
# /Project/Data/proc 
# /Project/Code 
# /Project/Code/Git-Repo # $CODEPATH 
# /Project/Code/Git-Repo/set_env_vars.sh

[ -z "$CODEPATH" ] && export CODEPATH=$(pwd)
[ -z "$DATAPATH" ] && export DATAPATH=$CODEPATH/data
[ -z "$RESULTSPATH" ] && export RESULTSPATH=$CODEPATH/results
mkdir -p $DATAPATH/raw
mkdir -p $DATAPATH/proc
mkdir -p $RESULTSPATH

# Set ports
# Exposing 3 ports: JuypyterLab, TensorBoard, Ray Dashboard
set_ports () {
for p in PORT1 PORT2 PORT3
do  
    unset $p
    case "$p" in 
    'PORT1')
    s=9000
    ;;
    'PORT2')
    s=6006
    ;;
    'PORT3')
    s=8265
    ;;
    esac
    echo "Setting port: $p" 
    
    local free=0
    until [ $free -gt 0 ]

    do  
        echo "Evaluating $s"
        check=$(netstat -ano | grep "$s") 
        # if a port is free, the netstat command return a positive exit code > 0
        if [ $? = 0 ]; then
        ((s ++))
        elif [ $? >  0 ]; then
        echo "$s is available"
        free=1
        fi
    done

    export "$p"="$s"
    echo "Set "$p" to "$s"."
done
}

set_ports