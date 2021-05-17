#!/usr/bin/env bash
export LOGLEVEL=DEBUG
WORKDIR=/path/to/lpot/examples/pytorch/image_recognition/imagenet/cpu/ptq

cd ${WORKDIR}

ARCH=resnet50

nohup python main.py \
    --pretrained \
    -a $ARCH \
    -b 30 \
    -e -t \
    --tuned_checkpoint ${ARCH}_basic_run \
    --conf ./conf.${ARCH}.basic.yaml \
    /data/dataset/imagenet/ilsvrc2012/torchvision 2>&1 | tee log.${ARCH}.basic &
