#!/bin/bash

tag=tvb/mrs-ci-rtx
ctnr=tvb-mrs-ci-rtx
work=/mnt/work/tvb-actions-runner

# these are the steps so it can be somewhat automated, but
# it's always finnicky to automate updating CI infra

docker stop $ctnr
docker rm $ctnr

docker build -t $tag -f mrs-ci-rtx.dockerfile .

docker run -d --name $ctnr --gpus all -v $work:/work $tag bash run.sh

git commit -am 'mrs-ci-rtx: update dockerfile & runner'
git push