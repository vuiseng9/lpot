#!/usr/bin/env bash
export LOGLEVEL=DEBUG
WORKDIR=/home/vchua/may17-lpot/lpot/examples/pytorch/image_recognition/imagenet/cpu/ptq

ARCH=resnet50

mkdir -p ${ARCH}_basic_run
cd ${WORKDIR}

nohup python main.py \
    --pretrained \
    -a $ARCH \
    -b 30 \
    -e -t \
    --tuned_checkpoint ${ARCH}_basic_run \
    --conf ./conf.${ARCH}.basic.yaml \
    /data/dataset/imagenet/ilsvrc2012/torchvision 2>&1 | tee ${ARCH}_basic_run/log.${ARCH}.basic &
